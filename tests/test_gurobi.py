import sys
import gurobipy as gp

def test_gurobi_model_creation():
    print("Python:", sys.executable)
    print("Gurobi:", gp.gurobi.version())
    m = gp.Model()
    assert m is not None
