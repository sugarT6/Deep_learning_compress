"""Training-prefix quality profiles and an explicit, target-free source mixture."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np

from .train import BalancedTrainingSampler, _write_json_atomic


PROFILE_FORMAT = "direct-quality-source-mixture-v1"
CYCLE_BINS = 10


def payload_sha256(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def cache_identity(sampler, dataset):
    reader = sampler.reader_for(dataset.accession)
    stat = reader.cache_path.stat()
    meta = reader.metadata
    return dict(source_sha256=meta.source_sha256, source_size=meta.source_size,
                read_count=meta.read_count, symbol_count=meta.symbol_count,
                train_stop=sampler.train_stop_for(dataset.accession),
                cache_size=stat.st_size, cache_mtime_ns=stat.st_mtime_ns)


def profile_ranges(stop, requested, strata):
    """Centered disjoint chunks, bounded strictly inside the training prefix."""
    if min(stop, requested, strata) <= 0:
        raise ValueError("profile sizes must be positive")
    count = min(requested, stop)
    groups = min(strata, count)
    quotas = np.full(groups, count // groups, dtype=np.int64)
    remaining = count % groups
    for index in range(groups):
        capacity = (index + 1) * stop // groups - index * stop // groups
        if remaining and quotas[index] < capacity:
            quotas[index] += 1
            remaining -= 1
    result = []
    for index in range(groups):
        left, right = index * stop // groups, (index + 1) * stop // groups
        quota = int(quotas[index])
        begin = left + (right - left - quota) // 2
        for start in range(begin, begin + quota, 256):
            result.append([start, min(start + 256, begin + quota)])
    return result


def summarize_quality_batches(batches):
    histogram = np.zeros(42, np.int64)
    changes = np.zeros(6, np.int64)
    cycle_count = np.zeros(CYCLE_BINS, np.int64)
    cycle_sum = np.zeros(CYCLE_BINS, np.float64)
    cycle_low = np.zeros(CYCLE_BINS, np.int64)
    reads = 0
    digest = hashlib.sha256()
    for batch in batches:
        q = batch.qualities.astype(np.int16)
        mask = batch.active_mask
        values = q[mask]
        if values.size and (values.min() < 0 or values.max() >= 42):
            raise ValueError("profile contains an invalid quality ID")
        histogram += np.bincount(values, minlength=42)
        positions = np.broadcast_to(np.arange(q.shape[1]), q.shape)
        bins = np.minimum(positions * CYCLE_BINS // np.maximum(batch.lengths[:, None], 1),
                          CYCLE_BINS - 1)[mask]
        cycle_count += np.bincount(bins, minlength=CYCLE_BINS)
        cycle_sum += np.bincount(bins, weights=values, minlength=CYCLE_BINS)
        cycle_low += np.bincount(bins, weights=(values < 20).astype(np.int64),
                                 minlength=CYCLE_BINS).astype(np.int64)
        # Cast before subtraction; uint8 differences wrap around at zero.
        delta = np.abs(q[:, 1:] - q[:, :-1])[mask[:, 1:] & mask[:, :-1]]
        change_bins = np.searchsorted([0, 1, 2, 4, 9], delta, side="left")
        changes += np.bincount(change_bins, minlength=6)
        reads += batch.read_count
        for row, length in enumerate(batch.lengths):
            digest.update(np.asarray([batch.read_indices[row], length], dtype="<i8").tobytes())
            digest.update(batch.qualities[row, :length].tobytes())
    total = int(histogram.sum())
    if not total:
        raise ValueError("profile has no quality symbols")
    p = histogram / total
    transition = changes / max(int(changes.sum()), 1)
    if not changes.sum():
        transition[0] = 1.
    mean = float(p @ np.arange(42))
    low = float(p[:20].sum())
    cycle_mean = np.divide(cycle_sum, cycle_count,
                           out=np.full(CYCLE_BINS, mean), where=cycle_count != 0)
    cycle_fraction = np.divide(cycle_low, cycle_count,
                               out=np.full(CYCLE_BINS, low), where=cycle_count != 0)
    # Fixed block weights, not fitted on validation/test data. Histogram and
    # transition distances are Hellinger-like; curve features are normalized.
    curve = np.r_[cycle_mean / 41., cycle_fraction]
    features = np.r_[np.sqrt(.5 * p), np.sqrt(.25 * transition),
                     math.sqrt(.25 / len(curve)) * curve]
    return dict(reads=reads, quality_symbols=total, histogram=histogram.tolist(),
                change_counts=changes.tolist(), cycle_counts=cycle_count.tolist(),
                cycle_mean=cycle_mean.tolist(), cycle_low20_fraction=cycle_fraction.tolist(),
                mean_quality=mean, entropy_bits=float(-(p[p > 0] * np.log2(p[p > 0])).sum()),
                features=features.tolist(), sample_sha256=digest.hexdigest())


def cluster_profiles(profiles, clusters):
    """Best deterministic farthest-first/Lloyd start; canonical group labels."""
    if type(clusters) is not int or clusters < 1 or not profiles:
        raise ValueError("clusters and profile set must be nonempty")
    names = sorted(profiles)
    x = np.asarray([profiles[name]["features"] for name in names], dtype=np.float64)
    if x.ndim != 2 or not np.isfinite(x).all():
        raise ValueError("invalid quality profile feature matrix")
    best = None
    # Twenty source profiles are small enough to try every source as a first
    # center. Select solely by within-profile SSE, never model/test scores.
    for first in range(len(x)):
        centers = [x[first].copy()]
        for _ in range(1, min(clusters, len(x))):
            distances = ((x[:, None] - np.asarray(centers)) ** 2).sum(2).min(1)
            if distances.max() <= 1e-20:
                break
            centers.append(x[int(np.argmax(distances))].copy())
        centers = np.asarray(centers)
        labels = None
        for _ in range(100):
            new = ((x[:, None] - centers) ** 2).sum(2).argmin(1)
            if labels is not None and np.array_equal(labels, new):
                break
            labels = new
            centers = np.stack([x[labels == index].mean(0) for index in sorted(set(map(int, labels)))])
        distances = ((x[:, None] - centers) ** 2).sum(2)
        labels = distances.argmin(1)
        groups = sorted([[name for name, label in zip(names, labels) if label == index]
                         for index in sorted(set(map(int, labels)))], key=lambda group: group[0])
        candidate = (float(distances.min(1).sum()), groups)
        if best is None or candidate < best:
            best = candidate
    return best[1]


def original_probabilities(datasets, family_weights):
    families = {d.platform_family for d in datasets}
    if set(family_weights) != families or any(type(v) is not int or v <= 0 for v in family_weights.values()):
        raise ValueError("family weights must cover the training families")
    total = sum(family_weights.values())
    return {d.accession: family_weights[d.platform_family] / total /
            sum(other.platform_family == d.platform_family for other in datasets) for d in datasets}


def mixture_probabilities(original, groups, fraction):
    """Half original/half mode balance, with final per-source boost <=2x.

    Water filling redistributes clipped mass in proportion to the remaining
    mode weights. No weight is selected using validation or model losses.
    """
    raw = {name: 1. / len(groups) / len(group) for group in groups for name in group}
    if set(raw) != set(original) or sum(map(len, groups)) != len(original):
        raise ValueError("quality groups must partition training sources")
    balanced = raw.copy()
    if fraction:
        cap = {name: original[name] * (2. - (1. - fraction)) / fraction for name in raw}
        remaining, mass = set(raw), 1.
        while remaining:
            denominator = sum(raw[name] for name in sorted(remaining))
            proposal = {name: mass * raw[name] / denominator for name in sorted(remaining)}
            clipped = [name for name in sorted(remaining) if proposal[name] > cap[name] + 1e-15]
            if not clipped:
                balanced.update(proposal)
                break
            for name in clipped:
                balanced[name] = cap[name]
                mass -= cap[name]
                remaining.remove(name)
    mixed = {name: (1 - fraction) * original[name] + fraction * balanced[name] for name in original}
    return raw, balanced, mixed


def prepare_profile(path, *, datasets, cache_dir, family_weights, train_fraction=.9,
                    profile_reads=4096, strata=16, clusters=4, mode_fraction=.5, seed=20261028,
                    progress=False):
    path = Path(path)
    if path.exists():
        raise FileExistsError("profile already exists; reuse it or choose a new output")
    if not math.isfinite(mode_fraction) or not 0 <= mode_fraction <= 1:
        raise ValueError("mode fraction must be in [0, 1]")
    if not datasets or any(not d.training_source for d in datasets):
        raise ValueError("quality profiles may use training sources only")
    with BalancedTrainingSampler(datasets, Path(cache_dir), train_fraction=train_fraction,
            batch_reads=256, seed=seed, family_weights=family_weights) as sampler:
        entries = {}
        for dataset in datasets:
            accession = dataset.accession
            if progress:
                import sys
                print("Profile training prefix: " + accession, file=sys.stderr, flush=True)
            ranges = profile_ranges(sampler.train_stop_for(accession), profile_reads, strata)
            reader = sampler.reader_for(accession)
            summary = summarize_quality_batches(reader.read_range(*pair) for pair in ranges)
            entries[accession] = dict(summary, ranges=ranges,
                                     cache_identity=cache_identity(sampler, dataset))
        groups = cluster_profiles(entries, clusters)
        original = original_probabilities(datasets, family_weights)
        raw, balanced, mixed = mixture_probabilities(original, groups, mode_fraction)
        payload = dict(format=PROFILE_FORMAT, training_order=[d.accession for d in datasets],
            train_fraction=train_fraction, family_weights=family_weights,
            profile_reads=profile_reads, strata=strata, requested_clusters=clusters,
            mode_fraction=mode_fraction, feature_rule="sqrt_Q_hist_0.5_delta_0.25_relative_cycle10_mean_low20_0.25",
            clustering="best_profile_SSE_of_all_source_farthest_first_Lloyd_starts_100_iterations",
            profiles=entries, groups=groups, original_probabilities=original,
            uncapped_mode_probabilities=raw, mode_balanced_probabilities=balanced,
            max_source_probability_ratio=2., quality_probabilities=mixed,
            scope="Training prefixes only. No target/validation/unseen labels; no model inputs or architecture changed.")
        payload["sha256"] = payload_sha256(payload)
        _write_json_atomic(path, payload)
    return payload


def load_profile(path, sampler):
    payload = json.loads(Path(path).read_text())
    digest = payload.pop("sha256", None)
    if digest != payload_sha256(payload) or payload.get("format") != PROFILE_FORMAT:
        raise ValueError("quality profile format or SHA mismatch")
    if payload["training_order"] != [d.accession for d in sampler.datasets] or payload["train_fraction"] != sampler.train_fraction:
        raise ValueError("quality profile training split/manifest mismatch")
    for dataset in sampler.datasets:
        if payload["profiles"][dataset.accession]["cache_identity"] != cache_identity(sampler, dataset):
            raise ValueError("quality profile cache identity mismatch: " + dataset.accession)
    payload["sha256"] = digest
    return payload


class QualityMixtureSampler(BalancedTrainingSampler):
    """Reuse the original per-file block permutations; only change source quotas."""

    def __init__(self, *args, probabilities, profile_sha256, **kwargs):
        self.probabilities = dict(probabilities)
        self.profile_sha256 = profile_sha256
        datasets = args[0] if args else kwargs["datasets"]
        if (set(probabilities) != {d.accession for d in datasets}
                or any(not math.isfinite(p) or p <= 0 for p in probabilities.values())
                or not math.isclose(sum(probabilities.values()), 1., abs_tol=1e-12)):
            raise ValueError("invalid quality source probabilities")
        super().__init__(*args, **kwargs)

    def _next_epoch(self):
        # Allocate each new slot to the greatest cumulative probability deficit,
        # then shuffle the schedule. Rounding debt carries across schedule blocks.
        counts = np.asarray([self._sample_counts[d.accession] for d in self.datasets], dtype=np.int64)
        p = np.asarray([self.probabilities[d.accession] for d in self.datasets])
        schedule = []
        for slot in range(self.steps_per_epoch):
            index = int(np.argmax((self._total_samples + slot + 1) * p - counts))
            schedule.append(index)
            counts[index] += 1
        self._rng.shuffle(schedule)
        self._schedule, self._schedule_cursor = schedule, 0
        self._epoch += 1

    def config_dict(self):
        result = super().config_dict()
        result.update(strategy="profile_bound_source_quotas", version=1,
                      probabilities=self.probabilities, profile_sha256=self.profile_sha256,
                      remainder_policy="cumulative_max_deficit_then_shuffle")
        return result
