import sys
import os
from pathlib import Path
DIR_HOME = Path(__file__).parent

import atexit
from dotenv import dotenv_values
from fastmcp import FastMCP
from psycopg_pool import ConnectionPool
from pydantic import Field
import rdkit
from rdkit import Chem
from rdkit.Chem import Descriptors
from typing import Annotated

import literature_search.search as search
import llm.llm as llm
import rag

env_config = dotenv_values(DIR_HOME / ".config" / "example.env")
if os.path.exists(DIR_HOME / ".config" / ".env"):
    env_config = dotenv_values(DIR_HOME / ".config" / ".env")

postgres_host = env_config["CHEMBIOTOX_HOST"]
postgres_port = env_config["CHEMBIOTOX_PORT"]
postgres_user = env_config["CHEMBIOTOX_USER"]
postgres_pass = env_config["CHEMBIOTOX_PASS"]

LLM = llm.create_llm_for_search()

DB_URI = f"postgresql://{postgres_user}:{postgres_pass}@{postgres_host}:{postgres_port}/chembiotox_v2"
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
def literature_search(query: Annotated[str, Field( description="Query to perform a PubMed literature search on", min_length=1, max_length=9999)]) -> str:
    """
    Given a query, return relevant academic and scientific papers from PubMed. Use this tool if the user requests a literature search.
    """
    response = search.scholar2result_llm(LLM, query=query)
    return response

@mcp.tool
def rag_search(query: Annotated[str, Field( description="Query to search across NTP publications", min_length=1, max_length=9999)]) -> str:
    """
    Given a query, return relevant toxicological information from publications from the National Toxicology Program (NTP) at https://ntp.niehs.nih.gov/publications. These reports are retrieved via retrieval-augmented generation (RAG). The publications include chemical, toxicity, and technical reports. This tool should be used if the user requests a literature search or a RAG search.
    """
    response = None
    try:
        response = rag.query(query, llm=LLM, use_training_data=True)
        used_rag_context = response['steps_taken'] and (response['steps_taken'][-1] == 'query_with_context')
        response = (f'*[The following response was taken from ' + ("RAG resources" if used_rag_context else "model's training knowledge") + ']*\n\n' + 
                    '**Response:**\n' + response['response'] + '\n\n' +
                    '**Searched Keyphrases:**\n' + '\n'.join([f'- {x}' for x in response['searched_keyphrases']]))
    except Exception as e:
        print("Error performing search.")
        print(e)
        return f"Error: query failed to run with message: {e}."
    
    return response

@mcp.tool
def smiles_to_mol_weight(smiles: Annotated[str, Field( description="SMILES string representing a chemical's structure", min_length=1, max_length=255)]) -> float:
    """
    Given a SMILES string, return the average molecular weight in g/mol of the chemical.
    """
    m = Chem.MolFromSmiles(smiles)
    wt = Descriptors.MolWt(m)
    return wt

@mcp.tool
def smiles_to_name(smiles: Annotated[str, Field( description="SMILES representation of a chemical", min_length=1, max_length=255)]) -> list[str]:
    """
    Given a chemical's SMILES representation, return its preferred name. If an exact mapping could not be found, the most structurally similar chemical's name is returned instead. Each result from this tool is structured as follows: chemical_name | tanimoto similarity
    """
    try:
        with pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(f"""
                SELECT DISTINCT
                    bcc.preferred_name,
                    bcrf.similarity
                FROM
                    base_chemicals bc,
                    base_chemical_compounds bcc,
                    base_chemical_to_smiles bcs,
                    get_morgan_fp_neighbors(%s) bcrf
                WHERE
                    bc.epa_id = bcc.epa_id
                AND bc.epa_id = bcs.epa_id
                AND bcrf.smi_id = bcs.smi_id
                AND bcrf.similarity > 0.7
                ORDER BY similarity DESC
                """, (smiles,))
                chemical_name = cur.fetchall()

                if chemical_name is None:
                    return "no chemical name obtained"
                
                chemical_names_formatted = []
                for chnm in chemical_name:
                    chemical_names_formatted.append(
                        f"{chnm[0]} | {chnm[1]}"
                    )
                return chemical_names_formatted
    except Exception as e:
        return f"Error fetching chemical name: {str(e)}"

@mcp.tool
def casrn_to_name(casrn: Annotated[str, Field( description="CASRN number for a chemical", min_length=1, max_length=255)]) -> str:
    """
    Given a chemical's CASRN, return its preferred name.
    """
    try:
        with pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(f"""
                SELECT DISTINCT
                    bcc.preferred_name
                FROM
                    base_chemical_compounds bcc
                WHERE
                    bcc.casrn = %s
                """, (casrn,))
                chemical_name = cur.fetchone()
                if chemical_name is None:
                    return "no chemical name obtained"
                return chemical_name[0]
    except Exception as e:
        return f"Error fetching chemical name: {str(e)}"

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

#@mcp.tool
#def leadscope_qsar_model_predictions(chemical_name: Annotated[str, Field( description="Preferred name of a chemical", min_length=1, max_length=255)]) -> list[str]:
#    """
#    Given the name of a chemical, return interpretations of its predicted toxicological properties based on Leadscope QSAR models.
#    """
#    try:
#        with pool.connection() as conn:
#            with conn.cursor() as cur:
#                cur.execute(f"""
#                SELECT DISTINCT
#                    lpm.model_name,
#                    lcp.model_value,
#                    lqd.endpoint_desc
#                FROM
#                    base_chemicals bc,
#                    base_chemical_to_pubchem_synonyms bpcs,
#                    base_chemical_to_smiles bcs,
#                    leadscope_chemical_predictions lcp,
#                    leadscope_qmrf_descriptions lqd,
#                    leadscope_predictive_models lpm
#                WHERE
#                    UPPER(bpcs.synonym) = UPPER(%s)
#                AND bc.epa_id = bpcs.epa_id
#                AND bc.epa_id = bcs.epa_id
#                AND bcs.smi_id = lcp.smi_id
#                AND lcp.lmodel_id = lpm.lmodel_id
#                AND lcp.lmodel_id = lqd.lmodel_id      
#                """, (chemical_name,))
#                interpretations = cur.fetchall()
#                if interpretations is None:
#                    return ["no Leadscope model data available"]
#                
#                interpretations_formatted = []
#                for interpretation in interpretations:
#                    interpretations_formatted.append(
#                        f"Model: {interpretation[0]}, Prediction: {interpretation[1]}, Description: {interpretation[2]}"
#                    )
#                return interpretations_formatted
#
#    except Exception as e:
#        return [f"Error fetching Leadscope models: {str(e)}"]

@mcp.tool
def ctd_chemical_to_genes(chemical_name: Annotated[str, Field( description="Preferred name of a chemical", min_length=1, max_length=255)]) -> list[str]:
    """
    Given the name of a chemical, return that chemical's associated gene interactions from the Comparative Toxicogenomics database (CTD).
    """
    try:
        with pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(f"""
                SELECT DISTINCT
                    ccg.interaction,
                    ccg.organism
                FROM
                    base_chemicals bc,
                    base_chemical_to_pubchem_synonyms bpcs,
                    base_chemical_to_smiles bcs,
                    ctd_to_base_chemicals cbc,
                    ctd_chemicals_to_genes ccg
                WHERE
                    UPPER(bpcs.synonym) = UPPER(%s)
                AND bc.epa_id = bpcs.epa_id
                AND bc.epa_id = bcs.epa_id
                AND bcs.smi_id = cbc.smi_id
                AND cbc.ctd_id = ccg.ctd_id
                """, (chemical_name,))
                interpretations = cur.fetchall()
                if interpretations is None:
                    return ["no CTD gene association data available"]
                
                interpretations_formatted = []
                for interpretation in interpretations:
                    interpretations_formatted.append(
                        f"Interaction: {interpretation[0]}, Organism: {interpretation[1]}"
                    )
                return interpretations_formatted

    except Exception as e:
        return [f"Error fetching CTD gene association data: {str(e)}"]
    
@mcp.tool
def ctd_chemical_to_diseases_direct(chemical_name: Annotated[str, Field( description="Preferred name of a chemical", min_length=1, max_length=255)]) -> list[str]:
    """
    Given the name of a chemical, return that chemical's associated diseases with direct evidence (i.e., from a marker) from the Comparative Toxicogenomics database (CTD).
    """
    try:
        with pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(f"""
                SELECT DISTINCT
                    ccd.disease_name
                FROM
                    base_chemicals bc,
                    base_chemical_to_pubchem_synonyms bpcs,
                    base_chemical_to_smiles bcs,
                    ctd_to_base_chemicals cbc,
                    ctd_chemicals_to_diseases ccd
                WHERE
                    UPPER(bpcs.synonym) = UPPER(%s)
                AND bc.epa_id = bpcs.epa_id
                AND bc.epa_id = bcs.epa_id
                AND bcs.smi_id = cbc.smi_id
                AND cbc.ctd_id = ccd.ctd_id
                AND ccd.direct_evidence IS NOT NULL
                """, (chemical_name,))
                interpretations = cur.fetchall()
                if interpretations is None:
                    return ["no CTD disease association data available"]
                
                interpretations_formatted = []
                for interpretation in interpretations:
                    interpretations_formatted.append(f"{interpretation[0]}")
                return interpretations_formatted

    except Exception as e:
        return [f"Error fetching CTD disease association data: {str(e)}"]
    
@mcp.tool
def ctd_chemical_to_diseases_inferred(chemical_name: Annotated[str, Field( description="Preferred name of a chemical", min_length=1, max_length=255)]) -> list[str]:
    """
    Given the name of a chemical, return that chemical's associated diseases with inferred evidence (i.e., from a gene) from the Comparative Toxicogenomics database (CTD). The output from this tool is a list of strings, with each string being of the format: disease_name | (gene from which the association was inferred from)
    """
    try:
        with pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(f"""
                SELECT DISTINCT
                    ccd.disease_name,
                    ccd.inference_score,
                    ccd.inference_gene_symbol
                FROM
                    base_chemicals bc,
                    base_chemical_to_pubchem_synonyms bpcs,
                    base_chemical_to_smiles bcs,
                    ctd_to_base_chemicals cbc,
                    ctd_chemicals_to_diseases ccd
                WHERE
                    UPPER(bpcs.synonym) = UPPER(%s)
                AND bc.epa_id = bpcs.epa_id
                AND bc.epa_id = bcs.epa_id
                AND bcs.smi_id = cbc.smi_id
                AND cbc.ctd_id = ccd.ctd_id
                AND ccd.direct_evidence IS NULL
                ORDER BY ccd.inference_score DESC
                LIMIT 10
                """, (chemical_name,))
                interpretations = cur.fetchall()
                if interpretations is None:
                    return ["no CTD disease association data available"]
                
                interpretations_formatted = []
                for interpretation in interpretations:
                    interpretations_formatted.append(f"{interpretation[0]} | ({interpretation[2]})")
                return interpretations_formatted

    except Exception as e:
        return [f"Error fetching CTD disease association data: {str(e)}"]

@mcp.tool
def ctd_chemical_to_go_biological_process(chemical_name: Annotated[str, Field( description="Preferred name of a chemical", min_length=1, max_length=255)]) -> list[str]:
    """
    Given the name of a chemical, return that chemical's associated biological process GO terms from the Comparative Toxicogenomics database (CTD).
    """
    try:
        with pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(f"""
                SELECT DISTINCT
                    ccg.go_term_name,
                    ccg.target_match_qty
                FROM
                    base_chemicals bc,
                    base_chemical_to_pubchem_synonyms bpcs,
                    base_chemical_to_smiles bcs,
                    ctd_to_base_chemicals cbc,
                    ctd_chemicals_to_goenrichment ccg
                WHERE
                    UPPER(bpcs.synonym) = UPPER(%s)
                AND bc.epa_id = bpcs.epa_id
                AND bc.epa_id = bcs.epa_id
                AND bcs.smi_id = cbc.smi_id
                AND cbc.ctd_id = ccg.ctd_id
                AND ccg.ontology = 'Biological Process'
                AND ccg.corrected_pvalue < 0.05
                AND highest_go_level > 3
                ORDER BY ccg.target_match_qty DESC
                LIMIT 20
                """, (chemical_name,))
                interpretations = cur.fetchall()
                if interpretations is None:
                    return ["no CTD biological process GO term data available"]
                
                interpretations_formatted = []
                for interpretation in interpretations:
                    interpretations_formatted.append(f"{interpretation[0]}")
                return interpretations_formatted

    except Exception as e:
        return [f"Error fetching CTD biological process GO term data: {str(e)}"]
    

@mcp.tool
def ctd_chemical_to_go_cellular_component(chemical_name: Annotated[str, Field( description="Preferred name of a chemical", min_length=1, max_length=255)]) -> list[str]:
    """
    Given the name of a chemical, return that chemical's associated cellular component GO terms from the Comparative Toxicogenomics database (CTD).
    """
    try:
        with pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(f"""
                SELECT DISTINCT
                    ccg.go_term_name,
                    ccg.target_match_qty
                FROM
                    base_chemicals bc,
                    base_chemical_to_pubchem_synonyms bpcs,
                    base_chemical_to_smiles bcs,
                    ctd_to_base_chemicals cbc,
                    ctd_chemicals_to_goenrichment ccg
                WHERE
                    UPPER(bpcs.synonym) = UPPER(%s)
                AND bc.epa_id = bpcs.epa_id
                AND bc.epa_id = bcs.epa_id
                AND bcs.smi_id = cbc.smi_id
                AND cbc.ctd_id = ccg.ctd_id
                AND ccg.ontology = 'Cellular Component'
                AND ccg.corrected_pvalue < 0.05
                AND highest_go_level > 3
                ORDER BY ccg.target_match_qty DESC
                LIMIT 20
                """, (chemical_name,))
                interpretations = cur.fetchall()
                if interpretations is None:
                    return ["no CTD cellular component GO term data available"]
                
                interpretations_formatted = []
                for interpretation in interpretations:
                    interpretations_formatted.append(f"{interpretation[0]}")
                return interpretations_formatted

    except Exception as e:
        return [f"Error fetching CTD cellular component GO term data: {str(e)}"]
    
@mcp.tool
def ctd_chemical_to_go_molecular_function(chemical_name: Annotated[str, Field( description="Preferred name of a chemical", min_length=1, max_length=255)]) -> list[str]:
    """
    Given the name of a chemical, return that chemical's associated molecular function GO terms from the Comparative Toxicogenomics database (CTD).
    """
    try:
        with pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(f"""
                SELECT DISTINCT
                    ccg.go_term_name,
                    ccg.target_match_qty
                FROM
                    base_chemicals bc,
                    base_chemical_to_pubchem_synonyms bpcs,
                    base_chemical_to_smiles bcs,
                    ctd_to_base_chemicals cbc,
                    ctd_chemicals_to_goenrichment ccg
                WHERE
                    UPPER(bpcs.synonym) = UPPER(%s)
                AND bc.epa_id = bpcs.epa_id
                AND bc.epa_id = bcs.epa_id
                AND bcs.smi_id = cbc.smi_id
                AND cbc.ctd_id = ccg.ctd_id
                AND ccg.ontology = 'Molecular Function'
                AND ccg.corrected_pvalue < 0.05
                AND highest_go_level > 3
                ORDER BY ccg.target_match_qty DESC
                LIMIT 20
                """, (chemical_name,))
                interpretations = cur.fetchall()
                if interpretations is None:
                    return ["no CTD molecular function GO term data available"]
                
                interpretations_formatted = []
                for interpretation in interpretations:
                    interpretations_formatted.append(f"{interpretation[0]}")
                return interpretations_formatted

    except Exception as e:
        return [f"Error fetching CTD molecular function GO term data: {str(e)}"]
    

@mcp.tool
def tox21_assay_predictions(chemical_name: Annotated[str, Field( description="Preferred name of a chemical", min_length=1, max_length=255)]) -> list[str]:
    """
    Given the name of a chemical, return that chemical's predicted behavior(s) from Tox21 assays. Each item in the output list contains the assay model name and if the chemical was predicted to be active or inactive.
    """
    try:
        with pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(f"""
                SELECT DISTINCT
                    tbc.assay_model,
                    tbc.activity_score
                FROM
                    base_chemicals bc,
                    base_chemical_to_pubchem_synonyms bpcs,
                    tox21_base_chemical_assay_predictions tbc
                WHERE
                    UPPER(bpcs.synonym) = UPPER(%s)
                AND bc.epa_id = bpcs.epa_id
                AND bc.epa_id = tbc.epa_id
                AND (tbc.activity_score >= 0.7 OR tbc.activity_score <= 0.3)
                ORDER BY tbc.activity_score DESC
                LIMIT 50
                """, (chemical_name,))
                interpretations = cur.fetchall()
                if interpretations is None:
                    return ["no Tox21 predictions available"]
                
                interpretations_formatted = []
                for interpretation in interpretations:
                    status = "inactive"
                    if interpretation[1] >= 0.7:
                        status = "active"
                    interpretations_formatted.append(f"{interpretation[0]}: {status}")
                return interpretations_formatted

    except Exception as e:
        return [f"Error fetching Tox21 predictions: {str(e)}"]
    

@mcp.tool
def drugbank_genes(chemical_name: Annotated[str, Field( description="Preferred name of a chemical", min_length=1, max_length=255)]) -> list[str]:
    """
    Given the name of a chemical, return that chemical's associated gene interactions as documented in DrugBank. This tool returns a list of strings, where each entry is structured like the following: gene_name | interaction
    """
    try:
        with pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(f"""
                SELECT DISTINCT
                    dcg.genename,
                    dcg.general_function
                FROM
                    base_chemicals bc,
                    base_chemical_to_pubchem_synonyms bpcs,
                    drugbank_to_base_chemicals dbc,
                    drugbank_curated_chemicals dcc,
                    drugbank_chemicals_to_genes dcg
                WHERE
                    UPPER(bpcs.synonym) = UPPER(%s)
                AND bc.epa_id = bpcs.epa_id
                AND bc.epa_id = dbc.epa_id
                AND dbc.drugbank_id = dcc.drugbank_id
                AND dcc.db_id = dcg.db_id
                """, (chemical_name,))
                interpretations = cur.fetchall()

                if interpretations is None:
                    return ["no gene interactions available"]
                
                interpretations_formatted = []
                for interpretation in interpretations:
                    interpretations_formatted.append(f"{interpretation[0]} | {interpretation[1]}")

                return interpretations_formatted

    except Exception as e:
        return [f"Error fetching gene interactions: {str(e)}"]

@mcp.tool
def drugbank_atccodes(chemical_name: Annotated[str, Field( description="Preferred name of a chemical", min_length=1, max_length=255)]) -> list[str]:
    """
    Given the name of a chemical, return that chemical's therapeutic properties as documented in DrugBank. Use this tool to retrieve a chemical's organ interactions, therapeutic use, and chemical properties.
    """
    try:
        with pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(f"""
                SELECT DISTINCT
                    dca.atc_annotation
                FROM
                    base_chemicals bc,
                    base_chemical_to_pubchem_synonyms bpcs,
                    drugbank_to_base_chemicals dbc,
                    drugbank_curated_chemicals dcc,
                    drugbank_chemicals_to_atccodes dca
                WHERE
                    UPPER(bpcs.synonym) = UPPER(%s)
                AND bc.epa_id = bpcs.epa_id
                AND bc.epa_id = dbc.epa_id
                AND dbc.drugbank_id = dcc.drugbank_id
                AND dcc.db_id = dca.db_id
                """, (chemical_name,))
                interpretations = cur.fetchall()
                if interpretations is None:
                    return ["no therapeutic properties available"]
                
                interpretations_formatted = []
                for interpretation in interpretations:
                    interpretations_formatted.append(f"{interpretation[0]}")
                return interpretations_formatted

    except Exception as e:
        return [f"Error fetching therapeutic properties: {str(e)}"]


@mcp.tool
def genra_results(chemical_name: Annotated[str, Field( description="Preferred name of a chemical", min_length=1, max_length=255)]) -> list[str]:
    """
    Given the name of a chemical, return corresponding predicted organ interactions, protein interactions, health effects, associated diseases and neoplasticity, developmental toxicity, chronic toxicity, sub-chronic toxicity, subacute toxicity, and reproductive toxicity from the EPA's Generalized Read Across (GenRA) tool.
    This tool outputs a list of strings. Each string is formatted as follows:

    category:subcategory - effect

    Where category may be one of the following:
    - SAC - Subacute Toxicity
    - MGR - Multigenerational Reproductive Toxicity
    - REP - Reproductive Toxicity
    - CHR - Chronic Toxicity
    - DEV - Developmental Toxicity
    - SUB - Sub-chronic Toxicity

    """
    try:
        with pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(f"""
                SELECT DISTINCT
                    gc.genra_category,
                    gc.genra_category_name,
                    bcg.genra_result
                FROM
                    base_chemicals bc,
                    base_chemical_to_pubchem_synonyms bpcs,
                    base_chemical_genra_results bcg,
                    genra_categories gc
                WHERE
                    UPPER(bpcs.synonym) = UPPER(%s)
                AND bc.epa_id = bpcs.epa_id
                AND bc.epa_id = bcg.epa_id
                AND bcg.gcat_id = gc.gcat_id
                AND bcg.genra_result != 'no_effect'
                AND bcg.genra_result != 'no_data'
                AND bcg.genra_result IS NOT NULL

                """, (chemical_name,))
                interpretations = cur.fetchall()
                if interpretations is None:
                    return ["no GenRA data available"]
                
                interpretations_formatted = []
                for interpretation in interpretations:
                    interpretations_formatted.append(f"{interpretation[0]}:{interpretation[1]} - {interpretation[2]}")
                return interpretations_formatted

    except Exception as e:
        return [f"Error fetching GenRA data: {str(e)}"]


def cleanup():
    pool.close()

atexit.register(cleanup)


if __name__ == "__main__":
    mcp.run()

