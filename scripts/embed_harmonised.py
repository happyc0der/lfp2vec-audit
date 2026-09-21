"""Frozen wav2vec2 embeddings of harmonised (<=100 Hz) inputs, on a fixed per-channel subsample.

The efficiency question: how far does the audio checkpoint get with nothing trained but a linear
head, once both labs' inputs are filtered alike? One forward pass, no backpropagation. The
subsample (25 of each channel's 100 chunks, seed 0) was fixed before any result existed; every
channel is kept, so the paper's per-channel post-processing still applies.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from lfpaudit.data.corpus import Corpus
from lfpaudit.device import pick_device
from lfpaudit.features.embed import embed_chunks

PER_CHANNEL = 25
LOWPASS_HZ = 100.0
OUT = Path("data/features/w2v2_frozen_lp100_sub")


def main() -> None:
    corpus = Corpus.load({"ibl": "data/stores/ibl", "allen": "data/stores/allen"})
    index = corpus.index
    rng = np.random.default_rng(0)
    keep = (
        index.groupby(["group", "channel"], sort=True)["chunk_id"]
        .apply(
            lambda ids: rng.choice(ids.to_numpy(), size=min(PER_CHANNEL, len(ids)), replace=False)
        )
        .explode()
        .astype(np.int64)
        .to_numpy()
    )
    keep = np.sort(keep)
    device = pick_device().device
    print(f"{len(keep)} chunks on {device}", flush=True)
    start = time.time()
    embeddings = embed_chunks(
        corpus.take, keep, fs=corpus.fs, device=device, lowpass_hz=LOWPASS_HZ, progress_every=100
    )
    seconds = time.time() - start
    OUT.mkdir(parents=True, exist_ok=True)
    np.save(OUT / "chunk_ids.npy", keep)
    np.save(OUT / "embeddings.npy", embeddings.astype(np.float16))
    (OUT / "meta.json").write_text(
        json.dumps(
            {
                "per_channel": PER_CHANNEL,
                "lowpass_hz": LOWPASS_HZ,
                "chunks": int(len(keep)),
                "seconds": round(seconds, 1),
                "chunks_per_second": round(len(keep) / seconds, 1),
                "device": device,
            },
            indent=2,
        )
    )
    print(f"done in {seconds / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
