import os
import time
from session_py import Session, ElementPlate
from session_py.obj import read_obj_polylines, pair_polylines


def main():
    base = os.path.dirname(os.path.abspath(__file__))
    t0 = time.perf_counter()

    # 1. Import + pair polylines
    polylines = read_obj_polylines(os.path.join(base, "..", "session_data", "annen_polylines.obj"))
    pairs = pair_polylines(polylines)
    t1 = time.perf_counter()

    # 2. Create session with elements
    session = Session("WoodComplete")
    g = session.add_group("Elements")
    for a, b in pairs:
        session.add_element(
            ElementPlate(polygon=polylines[a].get_points(),
                         polygon_top=polylines[b].get_points(),
                         name=f"plate_{a}"), g)
    t2 = time.perf_counter()

    # 3. Compute face-to-face contacts
    session.compute_face_to_face(5.0, 50.0)
    t3 = time.perf_counter()

    # 4. Save
    session.pb_dump(os.path.join(base, "..", "session_data", "WoodComplete.pb"))
    t4 = time.perf_counter()

    def ms(a, b):
        return (b - a) * 1000.0
    print(f"{len(polylines)} polylines -> {len(pairs)} elements -> {session.graph.number_of_edges()} joints")
    print(f"  import+pair: {ms(t0,t1):.0f}ms  elements: {ms(t1,t2):.0f}ms  "
          f"contacts: {ms(t2,t3):.0f}ms  save: {ms(t3,t4):.0f}ms  total: {ms(t0,t4):.0f}ms")


if __name__ == "__main__":
    main()
