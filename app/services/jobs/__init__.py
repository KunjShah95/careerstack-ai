"""Job source adapters.

Each adapter exposes the same coroutine signature (see base.JobSource) and
normalises its provider's payload into the common raw-posting dict that
discovery.normalise_posting() already consumes.
"""
