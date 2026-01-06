import atexit
from dotenv import dotenv_values
from fastmcp import FastMCP
import os
from pathlib import Path
from psycopg_pool import ConnectionPool
from pydantic import Field
import rdkit
from rdkit import Chem
from rdkit.Chem import Descriptors
from typing import Annotated

DIR_HOME = Path(__file__).parent
env_config = dotenv_values(DIR_HOME / ".config" / "example.env")
if os.path.exists(DIR_HOME / ".config" / ".env"):
    env_config = dotenv_values(DIR_HOME / ".config" / ".env")

postgres_host = env_config["CHEMBIOTOX_HOST"]
postgres_port = env_config["CHEMBIOTOX_PORT"]
postgres_user = env_config["CHEMBIOTOX_USER"]
postgres_pass = env_config["CHEMBIOTOX_PASS"]

DB_URI = f"postgresql://{postgres_user}:{postgres_pass}@{postgres_host}:{postgres_port}/chembiotox_v2"
print(DB_URI)
connection_kwargs = {
    "autocommit": True,
    "prepare_threshold": 0,
}


pool = ConnectionPool(conninfo=DB_URI, max_size=20, kwargs=connection_kwargs, open=True)
pool.wait()

mcp = FastMCP(
    name="ChemBioTox",
    instructions="This server provides data and functions relating to toxicological and chemical attribute data for over one million chemicals studied by the EPA."
)

@mcp.tool
def smiles_to_mol_weight(smiles: Annotated[str, Field( description="SMILES string representing a chemical's structure", min_length=1, max_length=255)]) -> float:
    """
    Given a SMILES string, return the average molecular weight in g/mol of the chemical.
    """
    m = Chem.MolFromSmiles(smiles)
    wt = Descriptors.MolWt(m)
    return wt

@mcp.tool
def name_to_canonical_smiles(chemical_name: Annotated[str, Field( description="Preferred name of a chemical", min_length=1, max_length=255)]) -> str:
    """
    Given the name of a chemical, return its canonical SMILES representation.
    """
    try:
        with pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(f"""
                SELECT DISTINCT
                    css.canonical_smiles
                FROM
                    base_chemicals bc,
                    base_chemical_to_pubchem_synonyms bpcs,
                    base_chemical_to_smiles bcs,
                    smiles_to_canonical_smiles scs,
                    canonical_smiles_strings css
                WHERE
                    UPPER(bpcs.synonym) = UPPER(%s)
                AND bc.epa_id = bpcs.epa_id
                AND bc.epa_id = bcs.epa_id
                AND bcs.smi_id = scs.smi_id
                AND scs.csm_id = css.csm_id
                """, (chemical_name,))
                smiles = cur.fetchone()
                if smiles is None:
                    return "no SMILES obtained"
                return smiles[0]
    except Exception as e:
        return f"Error fetching SMILES: {str(e)}"

@mcp.tool
def leadscope_qsar_model_predictions(chemical_name: Annotated[str, Field( description="Preferred name of a chemical", min_length=1, max_length=255)]) -> list[str]:
    """
    Given the name of a chemical, return interpretations of its predicted toxicological properties based on Leadscope QSAR models.
    """
    try:
        with pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(f"""
                SELECT DISTINCT
                    lpm.model_name,
                    lcp.model_value,
                    lqd.endpoint_desc
                FROM
                    base_chemicals bc,
                    base_chemical_to_pubchem_synonyms bpcs,
                    base_chemical_to_smiles bcs,
                    leadscope_chemical_predictions lcp,
                    leadscope_qmrf_descriptions lqd,
                    leadscope_predicted_models lpm, 
                WHERE
                    UPPER(bpcs.synonym) = UPPER(%s)
                AND bc.epa_id = bpcs.epa_id
                AND bc.epa_id = bcs.epa_id
                AND bcs.smi_id = lcp.smi_id
                AND lcp.lmodel_id = lpm.lmodel_id
                AND lcp.lmodel_id = lqd.lmodel_id      
                """, (chemical_name,))
                interpretations = cur.fetchall()
                if interpretations is None:
                    return "no Leadscope model data available"
                
                interpretations_formatted = []
                for interpretation in interpretations:
                    interpretations_formatted.append(
                        f"Model: {interpretation[0]}, Prediction: {interpretation[1]}, Description: {interpretation[2]}"
                    )
                return interpretations_formatted

    except Exception as e:
        return f"Error fetching Leadscope models: {str(e)}"

def cleanup():
    pool.close()

atexit.register(cleanup)


if __name__ == "__main__":
    mcp.run()
