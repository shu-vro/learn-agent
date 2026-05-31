from src.vector_store.qdrant_store import client


def fetch_from_qdrant_and_save_it_to_aws():
    collection_name = "my_collection"

    all_points = []
    offset = None

    while True:
        points, offset = client.scroll(
            collection_name=collection_name,
            limit=100,  # batch size
            offset=offset,
            with_payload=True,
            with_vectors=True,  # False if you don't need vectors
        )

        all_points.extend(points)

        if offset is None:
            break

    print(f"Total points: {len(all_points)}")


__all__ = ["fetch_from_qdrant_and_save_it_to_aws"]
