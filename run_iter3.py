import sys, json, hybrid_novelty as H
r = H.run(sys.argv[1], threads=1)
print(json.dumps(r, indent=1), flush=True)
