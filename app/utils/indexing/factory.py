# app/utils/indexing/factory.py
from app.utils.indexing.linear_index import LinearIndex
from app.utils.indexing.kdtree_index import KDTreeIndex
from app.utils.indexing.clustered_index import ClusteredIndex
from app.utils.indexing.index_type import IndexType
from app.utils.indexing.base import Indexer

def create_index_by_type(index_type: IndexType) -> Indexer:
    if index_type == IndexType.LINEAR:
        return LinearIndex()
    elif index_type == IndexType.KDTREE:
        return KDTreeIndex()
    elif index_type == IndexType.CLUSTERED:
        return ClusteredIndex()
    else:
        raise ValueError(f"Unsupported index type: {index_type}")
