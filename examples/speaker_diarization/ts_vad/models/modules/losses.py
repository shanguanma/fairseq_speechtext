import torch
import logging
import math
import numpy as np

import fast_bss_eval


EPS = torch.finfo(torch.get_default_dtype()).eps


def log_power(source, cal_way="np"):
    if cal_way == "np":
        ratio = np.sum(source**2, axis=-1)
        sdr = 10 * np.log10(ratio / source.shape[-1] * 16000 + 1e-8)
    else:
        ratio = torch.sum(source**2, axis=-1)
        sdr = 10 * torch.log10(ratio / source.shape[-1] * 16000 + 1e-8)

    return sdr


class SDRLoss(torch.nn.Module):
    def __init__(
        self,
        eps=EPS,
    ):
        super().__init__()
        self.eps = float(eps)

    def forward(
        self,
        ref: torch.Tensor,
        est: torch.Tensor,
    ) -> torch.Tensor:
        """SDR forward.

        Args:
            ref: Tensor, (..., n_samples)
                reference signal
            est: Tensor (..., n_samples)
                estimated signal

        Returns:
            loss: (...,)
                the SDR loss (negative sdr)
        """

        noise = ref - est
        ratio = torch.sum(ref**2, axis=-1) / (torch.sum(noise**2, axis=-1) + self.eps)
        sdr = 10 * torch.log10(ratio + self.eps)

        return -1 * sdr


class SISNRLoss(torch.nn.Module):
    """SI-SNR (or named SI-SDR) loss

    A more stable SI-SNR loss with clamp from `fast_bss_eval`.

    Attributes:
        clamp_db: float
            clamp the output value in  [-clamp_db, clamp_db]
        zero_mean: bool
            When set to True, the mean of all signals is subtracted prior.
        eps: float
            Deprecated. Kept for compatibility.
    """

    def __init__(
        self,
        clamp_db=None,
        zero_mean=True,
        eps=None,
    ):
        super().__init__()

        self.clamp_db = clamp_db
        self.zero_mean = zero_mean
        if eps is not None:
            logging.warning("Eps is deprecated in si_snr loss, set clamp_db instead.")
            if self.clamp_db is None:
                self.clamp_db = -math.log10(eps / (1 - eps)) * 10

    def forward(self, ref: torch.Tensor, est: torch.Tensor) -> torch.Tensor:
        """SI-SNR forward.

        Args:

            ref: Tensor, (..., n_samples)
                reference signal
            est: Tensor (..., n_samples)
                estimated signal

        Returns:
            loss: (...,)
                the SI-SDR loss (negative si-sdr)
        """
        assert torch.is_tensor(est) and torch.is_tensor(ref), est

        si_snr = fast_bss_eval.si_sdr_loss(
            est=est,
            ref=ref,
            zero_mean=self.zero_mean,
            clamp_db=self.clamp_db,
            pairwise=False,
        )

        return si_snr


class wSDRLoss(torch.nn.Module):
    """SI-SNR (or named SI-SDR) loss

    A more stable SI-SNR loss with clamp from `fast_bss_eval`.

    Attributes:
        clamp_db: float
            clamp the output value in  [-clamp_db, clamp_db]
        zero_mean: bool
            When set to True, the mean of all signals is subtracted prior.
        eps: float
            Deprecated. Kept for compatibility.
    """

    def __init__(
        self,
        eps=None,
    ):
        super().__init__()
        self.eps = eps

    def forward(
        self, ref: torch.Tensor, est: torch.Tensor, mixture: torch.Tensor
    ) -> torch.Tensor:
        def bsum(x):
            return torch.sum(x, dim=-1)

        def mSDRLoss(orig, est):
            correlation = bsum(orig * est)
            energies = torch.norm(orig, p=2, dim=-1) * torch.norm(est, p=2, dim=-1)
            return -(correlation / (energies + EPS))

        noise = mixture - ref
        noise_est = mixture - est

        a = bsum(ref**2) / (bsum(ref**2) + bsum(noise**2) + EPS)

        wSDR = a * mSDRLoss(ref, est) + (1 - a) * mSDRLoss(noise, noise_est)
        return torch.mean(wSDR)


def log_power(source, cal_way="np"):
    if cal_way == "np":
        ratio = np.sum(source**2, axis=-1)
        sdr = 10 * np.log10(ratio / source.shape[-1] * 16000 + 1e-8)
    else:
        ratio = torch.sum(source**2, axis=-1)
        sdr = 10 * torch.log10(ratio / source.shape[-1] * 16000 + 1e-8)

    return sdr
