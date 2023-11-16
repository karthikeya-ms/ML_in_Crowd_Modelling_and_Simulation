mkdir -p VadereProject/Task1/output

scenario-customizer -s \
    -f "VadereProject/Task1/scenarios/Scenario 6.scenario" \
    -o "VadereProject/Task1/scenarios/Scenario 6 - Customized.scenario" \
    -p "11.50,1.50,24"

java -jar vadere/VadereSimulator/target/vadere-console.jar scenario-run \
    --scenario-file "VadereProject/Task1/scenarios/Scenario 6 - Customized.scenario" \
    --output-dir "VadereProject/Task1/output"
