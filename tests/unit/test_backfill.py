"""Unit tests for the backfill date chunker (pure logic, no DB, no network)."""

from datetime import date, timedelta

from volatility_mlops.ingestion.backfill import date_chunks


def test_chunks_tile_the_range_without_gaps_or_overlaps():
    start, end = date(2010, 1, 1), date(2012, 12, 31)
    chunks = date_chunks(start, end)

    assert chunks[0][0] == start
    assert chunks[-1][1] == end
    # Each chunk begins the day after the previous one ends: contiguous, no overlap.
    # Lengths differ by one by design, so strict=False is intentional here.
    for (_, prev_end), (next_start, _) in zip(chunks, chunks[1:], strict=False):
        assert next_start == prev_end + timedelta(days=1)


def test_yearly_width():
    chunks = date_chunks(date(2010, 1, 1), date(2012, 12, 31))
    assert len(chunks) == 3
    assert chunks[0] == (date(2010, 1, 1), date(2010, 12, 31))


def test_partial_final_chunk_is_clamped_to_end():
    chunks = date_chunks(date(2010, 1, 1), date(2010, 6, 15))
    assert chunks == [(date(2010, 1, 1), date(2010, 6, 15))]


def test_empty_when_start_after_end():
    assert date_chunks(date(2020, 1, 1), date(2019, 1, 1)) == []
