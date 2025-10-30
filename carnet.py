from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.inference import VariableElimination
from pgmpy.factors.discrete import TabularCPD

car_model = DiscreteBayesianNetwork(
    [
        ("Battery", "Radio"),
        ("Battery", "Ignition"),
        ("Ignition","Starts"),
        ("Gas","Starts"),
        ("Starts","Moves"),
])

# Defining the parameters using CPT


cpd_battery = TabularCPD(
    variable="Battery", variable_card=2, values=[[0.70], [0.30]],
    state_names={"Battery":['Works',"Doesn't work"]},
)

cpd_gas = TabularCPD(
    variable="Gas", variable_card=2, values=[[0.40], [0.60]],
    state_names={"Gas":['Full',"Empty"]},
)

cpd_radio = TabularCPD(
    variable=  "Radio", variable_card=2,
    values=[[0.75, 0.01],[0.25, 0.99]],
    evidence=["Battery"],
    evidence_card=[2],
    state_names={"Radio": ["turns on", "Doesn't turn on"],
                 "Battery": ['Works',"Doesn't work"]}
)

cpd_ignition = TabularCPD(
    variable=  "Ignition", variable_card=2,
    values=[[0.75, 0.01],[0.25, 0.99]],
    evidence=["Battery"],
    evidence_card=[2],
    state_names={"Ignition": ["Works", "Doesn't work"],
                 "Battery": ['Works',"Doesn't work"]}
)

cpd_starts = TabularCPD(
    variable="Starts",
    variable_card=2,
    values=[[0.95, 0.05, 0.05, 0.001], [0.05, 0.95, 0.95, 0.9999]],
    evidence=["Ignition", "Gas"],
    evidence_card=[2, 2],
    state_names={"Starts":['yes','no'], "Ignition":["Works", "Doesn't work"], "Gas":['Full',"Empty"]},
)

cpd_moves = TabularCPD(
    variable="Moves", variable_card=2,
    values=[[0.8, 0.01],[0.2, 0.99]],
    evidence=["Starts"],
    evidence_card=[2],
    state_names={"Moves": ["yes", "no"],
                 "Starts": ['yes', 'no'] }
)


# Associating the parameters with the model structure
car_model.add_cpds( cpd_starts, cpd_ignition, cpd_gas, cpd_radio, cpd_battery, cpd_moves)


# Step 3: Add KeyPresent -> Starts
cpd_key = TabularCPD(
    variable="KeyPresent", variable_card=2,
    values=[[0.7], [0.3]],
    state_names={"KeyPresent": ["yes", "no"]},
)

car_model.remove_cpds(cpd_starts)

cpd_starts = TabularCPD(
    variable="Starts", variable_card=2,
    values=[
        [0.99, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01],
        [0.01, 0.99, 0.99, 0.99, 0.99, 0.99, 0.99, 0.99],
    ],
    evidence=["Ignition", "Gas", "KeyPresent"],
    evidence_card=[2, 2, 2],
    state_names={
        "Starts": ["yes", "no"],
        "Ignition": ["Works", "Doesn't work"],
        "Gas": ["Full", "Empty"],
        "KeyPresent": ["yes", "no"],
    },
)

car_model.add_node("KeyPresent")
car_model.add_edge("KeyPresent", "Starts")
car_model.add_cpds(cpd_key, cpd_starts)
car_model.check_model()



car_infer = VariableElimination(car_model)

print(car_infer.query(variables=["Moves"],evidence={"Radio":"turns on", "Starts":"yes"}))

if __name__ == "__main__":
    infer = VariableElimination(car_model)

    # 1) P(Battery doesn't work | Moves = no)
    q1 = infer.query(variables=["Battery"], evidence={"Moves": "no"})
    print("1) P(Battery bad | Moves=no):", round(float(q1.values[1]), 4))

    # 2) P(Starts = no | Radio doesn't turn on)
    q2 = infer.query(variables=["Starts"], evidence={"Radio": "Doesn't turn on"})
    print("2) P(Starts=no | Radio=off):", round(float(q2.values[1]), 4))

    # 3) Radio on with Battery=Works vs also knowing Gas=Full
    q3a = infer.query(variables=["Radio"], evidence={"Battery": "Works"})
    q3b = infer.query(variables=["Radio"], evidence={"Battery": "Works", "Gas": "Full"})
    print("3) P(Radio=on | Battery=Works):", round(float(q3a.values[0]), 4))
    print("   P(Radio=on | Battery=Works, Gas=Full):", round(float(q3b.values[0]), 4))

    # 4) Ignition bad with Moves=no vs also Gas=Empty
    q4a = infer.query(variables=["Ignition"], evidence={"Moves": "no"})
    q4b = infer.query(variables=["Ignition"], evidence={"Moves": "no", "Gas": "Empty"})
    print("4) P(Ignition bad | Moves=no):", round(float(q4a.values[1]), 4))
    print("   P(Ignition bad | Moves=no, Gas=Empty):", round(float(q4b.values[1]), 4))

    # 5) P(Starts = yes | Radio=on, Gas=Full)
    q5 = infer.query(variables=["Starts"], evidence={"Radio": "turns on", "Gas": "Full"})
    print("5) P(Starts=yes | Radio=on, Gas=Full):", round(float(q5.values[0]), 4))

    # 6) Step 3 query: P(KeyPresent = no | Moves = no)
    q6 = infer.query(variables=["KeyPresent"], evidence={"Moves": "no"})
    print("6) P(KeyPresent=no | Moves=no):", round(float(q6.values[1]), 4))