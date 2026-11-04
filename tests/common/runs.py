"""
Holds `computed`, which names the nodes a graph's most recent run computed,
for the cases reading what a rerun takes back from the cache.
"""

from hamilton.caching.adapter import CachingEventType

from thermur.pipeline.graph import Graph


def computed(graph: Graph) -> set[str]:
    """
    Names each node the graph's most recent run computed rather than read
    back from its cache.
    """
    cache = graph.driver.cache
    return {
        node
        for node, events in cache.logs(cache.last_run_id, level="debug").items()
        if any(event.event_type is CachingEventType.EXECUTE_NODE for event in events)
    }
