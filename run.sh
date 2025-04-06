rm -rf chroma/company_vd/*
rm -rf chroma/rfp_vd/*
rm -rf *.json

python3 src/agents/eligibility_agent.py
python3 src/agents/contract_risk_agent.py
python3 src/agents/complaince_agent.py