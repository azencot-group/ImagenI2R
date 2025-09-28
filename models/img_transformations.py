from abc import ABC, abstractmethod
import torch

class TsImgEmbedder(ABC):
    """
    Abstract class for transforming time series to images and vice versa
    """

    def __init__(self, device, seq_len):
        self.device = device
        self.seq_len = seq_len

    @abstractmethod
    def ts_to_img(self, signal):
        """

        Args:
            signal: given time series

        Returns:
            image representation of the signal

        """
        pass

    @abstractmethod
    def img_to_ts(self, img):
        """

        Args:
            img: given generated image

        Returns:
            time series representation of the generated image
        """
        pass


class DelayEmbedder(TsImgEmbedder):
    """
    Delay embedding transformation
    """

    def __init__(self, device, seq_len, delay, embedding, batch_size, num_features):
        super().__init__(device, seq_len)
        self.device = device
        self.seq_len = seq_len
        self.delay = delay
        self.embedding = embedding
        self.batch_size = batch_size
        self.num_features = num_features
        self.img_shape = None
        self.mapping = None  # Mapping from TS positions to image positions

        # Create the helper series and image to build the mapping
        self.create_mapping()


    def pad_to_square(self, x, mask=0):
        """
        Pads the input tensor x to make it square along the last two dimensions.
        """
        _, _, cols, rows = x.shape
        max_side = max(cols, rows)
        padding = (
            0, max_side - rows, 0, max_side - cols)  # Padding format: (pad_left, pad_right, pad_top, pad_bottom)

        # Padding the last two dimensions to make them square
        x_padded = torch.nn.functional.pad(x, padding, mode='constant', value=mask)
        return x_padded

    def unpad(self, x, original_shape):
        """
        Removes the padding from the tensor x to get back to its original shape.
        """
        _, _, original_cols, original_rows = original_shape
        return x[:, :, :original_cols, :original_rows]

    def ts_to_img(self, signal, pad=True, mask=0):

        batch, length, features = signal.shape
        #  if our sequences are of different lengths, this can happen with physionet and climate datasets
        if self.seq_len != length:
            self.seq_len = length

        x_image = torch.zeros((batch, features, self.embedding, self.embedding))
        i = 0
        while (i * self.delay + self.embedding) <= self.seq_len:
            start = i * self.delay
            end = start + self.embedding
            x_image[:, :, :, i] = signal[:, start:end].permute(0, 2, 1)
            i += 1

        ### SPECIAL CASE
        if i * self.delay != self.seq_len and i * self.delay + self.embedding > self.seq_len:
            start = i * self.delay
            end = signal[:, start:].permute(0, 2, 1).shape[-1]
            x_image[:, :, :end, i] = signal[:, start:].permute(0, 2, 1)
            i += 1

        # cache the shape of the image before padding
        self.img_shape = (batch, features, self.embedding, i)
        x_image = x_image.to(self.device)[:, :, :, :i]

        if pad:
            x_image = self.pad_to_square(x_image, mask)

        return x_image


    def create_mapping(self):
        """
        Creates the mapping from time series positions to image positions.
        """
        # Create the helper series
        helper_series = torch.arange(1, self.seq_len + 1, dtype=torch.float32, device=self.device)
        helper_series = helper_series.unsqueeze(0).unsqueeze(-1)  # Shape: (1, seq_len, 1)
        helper_series = helper_series.repeat(self.batch_size, 1, self.num_features)  # Shape: (batch_size, seq_len, num_features)

        # Convert the helper series to image
        helper_image = self.ts_to_img(helper_series, pad=True, mask=0)

        # Unpad the helper image to get the original shape
        helper_image_non_square = self.unpad(helper_image, self.img_shape)
        batch_size, channels, rows, cols = helper_image_non_square.shape

        # Create the mapping
        self.mapping = {}
        # Since the helper series contains unique values from 1 to seq_len, and is the same across batch and features,
        # we can use the first sample and first feature to create the mapping
        for row in range(rows):
            for col in range(cols):
                val = helper_image_non_square[0, 0, row, col].item()
                if val != 0:  # Ignore padding or mask values
                    ts_idx = int(val) - 1  # Convert value back to TS index (0-based)
                    if ts_idx not in self.mapping:
                        self.mapping[ts_idx] = []
                    self.mapping[ts_idx].append((row, col))

    def img_to_ts(self, x_image_square):
        """
        Reconstructs the time series from its image representation.
        """
        # Unpad the image to get back to original shape
        x_image_non_square = self.unpad(x_image_square, self.img_shape)
        batch_size, channels, rows, cols = x_image_non_square.shape

        # Initialize the reconstructed TS tensor
        reconstructed_ts = torch.zeros((batch_size, self.seq_len, channels), device=self.device)

        # Use the mapping to reconstruct the TS
        for ts_idx in range(self.seq_len):
            image_positions = self.mapping.get(ts_idx, [])
            if not image_positions:
                continue  # Skip if there are no corresponding image positions

            values = []
            for row, col in image_positions:
                value = x_image_non_square[:, :, row, col]
                values.append(value)

            # Stack and average the values
            values_tensor = torch.stack(values, dim=0)
            mean_values = values_tensor.mean(dim=0)
            reconstructed_ts[:, ts_idx, :] = mean_values

        # Permute to get shape (batch_size, seq_len, channels)
        reconstructed_ts = reconstructed_ts.permute(0, 1, 2)

        return reconstructed_ts.cuda()
