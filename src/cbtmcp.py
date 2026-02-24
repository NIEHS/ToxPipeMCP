import os
from pathlib import Path
DIR_HOME = Path(__file__).parent

from dotenv import dotenv_values
from fastmcp import FastMCP
from pydantic import Field
from rdkit import Chem
from rdkit.Chem import Descriptors
from typing import Annotated

import literature_search.search as search
import llm.llm as llm
import rag

import urllib.parse
import requests
import json

env_config = dotenv_values(DIR_HOME / ".config" / "example.env")
if os.path.exists(DIR_HOME / ".config" / ".env"):
    env_config = dotenv_values(DIR_HOME / ".config" / ".env")

CHEMBIOTOX_URL = env_config["CHEMBIOTOX_URL"]

LLM = llm.create_llm_for_search()

mcp = FastMCP(
    name="ChemBioTox",
    instructions="This server provides data and functions relating to toxicological and chemical attribute data for over one million chemicals studied by the EPA."
)

@mcp.tool
def literature_search(query: Annotated[str, Field( description="Query to perform a PubMed literature search on", min_length=1, max_length=9999)]) -> str:
    """Given a query, return relevant academic and scientific papers from PubMed. Use this tool if the user requests a literature search.

    Args:
        query: The query to perform a literature search on. This should be a string of at least one character and at most 9999 characters. The query should be specific enough to yield relevant results.
    Returns:
        A string containing the results of the literature search.
    """
    response = search.scholar2result_llm(LLM, query=query)
    return response

@mcp.tool
def rag_search(query: Annotated[str, Field( description="Query to search across NTP publications", min_length=1, max_length=9999)]) -> str:
    """
    Given a query, return relevant toxicological information from publications from the National Toxicology Program (NTP) at https://ntp.niehs.nih.gov/publications. These reports are retrieved via retrieval-augmented generation (RAG). The publications include chemical, toxicity, and technical reports. This tool should be used if the user requests a literature search or a RAG search.

    Args:
        query: The query to perform a RAG search on. This should be a string of at least one character and at most 9999 characters. The query should be specific enough to yield relevant results.
    Returns:
        A string containing the results of the RAG search, which may include information from the model's training data and/or from NTP publications.
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
def is_valid_smiles(smiles: Annotated[str, Field( description="SMILES string representing a chemical's structure", min_length=1, max_length=255)]) -> bool:
    """
    Given a SMILES string, return whether or not it is a valid SMILES representation.

    Args:
        smiles: A SMILES string representing a chemical's structure. This should be a string of at least one character and at most 255 characters.
    Returns:
        A boolean indicating whether or not the given SMILES string is valid.
    """
    m = Chem.MolFromSmiles(smiles)
    return m is not None

@mcp.tool
def smiles_to_mol_weight(smiles: Annotated[str, Field( description="SMILES string representing a chemical's structure", min_length=1, max_length=255)]) -> float:
    """
    Given a SMILES string, return the average molecular weight in g/mol of the chemical.

    Args:
        smiles: A SMILES string representing a chemical's structure. This should be a string of at least one character and at most 255 characters.
    Returns:
        The average molecular weight in g/mol of the chemical.
    """
    m = Chem.MolFromSmiles(smiles)
    wt = Descriptors.MolWt(m)
    return wt

@mcp.tool
def smiles_to_name(smiles: Annotated[str, Field( description="SMILES representation of a chemical", min_length=1, max_length=255)]) -> list[str]:
    """
    Given a chemical's SMILES representation, return its preferred name. If an exact mapping could not be found, the most structurally similar chemical's name is returned instead.

    Args:
        smiles: A SMILES string representing a chemical's structure. This should be a string of at least one character and at most 255 characters.
    Returns:
        A list of strings, where each string is structured as follows: chemical_name | tanimoto similarity. The list is ordered from most to least similar chemical, and only chemicals with a Tanimoto similarity of 0.8 or higher are included in the output. If no chemicals with a Tanimoto similarity of 0.8 or higher are found, an empty list is returned.
    """
    try:
        params = {
            'smiles': smiles
        }
        encoded_params = urllib.parse.urlencode(params)

        r = requests.get(f"{CHEMBIOTOX_URL}mcp/smiles_to_name?{encoded_params}")
        js = json.loads(r.text)
        
        out = []
        for i in js:
            out.append(f"{i['preferred_name']} | {i['similarity']}")
        return out
    
    except Exception as e:
        return [""]

@mcp.tool
def casrn_to_name(casrn: Annotated[str, Field( description="CASRN number for a chemical", min_length=1, max_length=255)]) -> str:
    """
    Given a chemical's CASRN, return its preferred name.

    Args:
        casrn: A string representing the CASRN number for a chemical. This should be a string of at least one character and at most 255 characters.
    Returns:
        A string representing the preferred name of the chemical corresponding to the given CASRN. If no chemical with the given CASRN is found, a string indicating that no chemical name could be obtained is returned instead.
    """
    try:
        params = {
            'casrn': casrn
        }
        encoded_params = urllib.parse.urlencode(params)
        r = requests.get(f"{CHEMBIOTOX_URL}mcp/casrn_to_name?{encoded_params}")
        js = json.loads(r.text)
        
        out = js[0]['preferred_name']
        return out
    
    except Exception as e:
        return ""

@mcp.tool
def name_to_canonical_smiles(chemical_name: Annotated[str, Field( description="Preferred name of a chemical", min_length=1, max_length=255)]) -> str:
    """
    Given the name of a chemical, return its canonical SMILES representation.

    Args:
        chemical_name: A string representing the preferred name of a chemical. This should be a string of at least one character and at most 255 characters.
    Returns:
        A string representing the canonical SMILES representation of the chemical corresponding to the given name. If no chemical with the given name is found, a string indicating that no SMILES could be obtained is returned instead.
    """
    try:
        params = {
            'chemical_name': chemical_name
        }
        encoded_params = urllib.parse.urlencode(params)
        r = requests.get(f"{CHEMBIOTOX_URL}mcp/name_to_canonical_smiles?{encoded_params}")
        js = json.loads(r.text)
        
        out = js[0]['canonical_smiles']
        return out
    
    except Exception as e:
        return ""

@mcp.tool
def ctd_chemical_to_genes(chemical_name: Annotated[str, Field( description="Preferred name of a chemical", min_length=1, max_length=255)], species: Annotated[str, Field( description="Species corresponding to genes. Must be exactly one of: Homo sapiens, Mus musculus, Rattus norvegicus", min_length=1, max_length=255)]="Homo sapiens") -> list[str]:
    """
    Given the name of a chemical and a species, return that chemical's associated gene interactions from the Comparative Toxicogenomics database (CTD). This tool returns a list of strings, where each string is an interaction for the given chemical in the specified species passed to the tool.

    Args:
        chemical_name: A string representing the preferred name of a chemical. This should be a string of at least one character and at most 255 characters.
        species: A string representing the species for which gene interactions are requested. Must be exactly one of: Homo sapiens, Mus musculus, Rattus norvegicus.
    Returns:
        A list of strings, where each string is an interaction for the given chemical in the specified species. If no interactions are found for the given chemical in the specified species, an empty list is returned.
    """
    try:
        params = {
            'chemical_name': chemical_name,
            'species': species
        }
        encoded_params = urllib.parse.urlencode(params)
        r = requests.get(f"{CHEMBIOTOX_URL}mcp/ctd_chemical_to_genes?{encoded_params}")
        js = json.loads(r.text)
        
        out = []
        for i in js:
            out.append(i['interaction'])
        return out
    
    except Exception:
        return [""]
    
@mcp.tool
def ctd_chemical_to_diseases_direct(chemical_name: Annotated[str, Field( description="Preferred name of a chemical", min_length=1, max_length=255)]) -> list[str]:
    """
    Given the name of a chemical, return that chemical's associated diseases with direct evidence (i.e., from a marker) from the Comparative Toxicogenomics database (CTD).

    Args:
        chemical_name: A string representing the preferred name of a chemical. This should be a string of at least one character and at most 255 characters.
    Returns:
        A list of strings, where each string is a disease associated with the given chemical with direct evidence in CTD. If no diseases with direct evidence associations are found for the given chemical, an empty list is returned.
    """
    try:
        params = {
            'chemical_name': chemical_name
        }
        encoded_params = urllib.parse.urlencode(params)
        r = requests.get(f"{CHEMBIOTOX_URL}mcp/ctd_chemical_to_diseases_direct?{encoded_params}")
        js = json.loads(r.text)
        
        out = []
        for i in js:
            out.append(i['disease_name'])
        return out
    
    except Exception:
        return [""]
    
@mcp.tool
def ctd_chemical_to_diseases_inferred(chemical_name: Annotated[str, Field( description="Preferred name of a chemical", min_length=1, max_length=255)]) -> list[str]:
    """
    Given the name of a chemical, return that chemical's associated diseases with inferred evidence (i.e., from a gene) from the Comparative Toxicogenomics database (CTD). The output from this tool is a list of strings, with each string being of the format: disease_name | inference_score | gene from which the association was inferred

    Args:
        chemical_name: A string representing the preferred name of a chemical. This should be a string of at least one character and at most 255 characters.
    Returns:
        A list of strings, with each string being of the format: disease_name | inference_score | gene from which the association was inferred. If no diseases with inferred evidence associations are found for the given chemical, an empty list is returned.
    """
    try:
        params = {
            'chemical_name': chemical_name
        }
        encoded_params = urllib.parse.urlencode(params)
        r = requests.get(f"{CHEMBIOTOX_URL}mcp/ctd_chemical_to_diseases_inferred?{encoded_params}")
        js = json.loads(r.text)
        
        out = []
        for i in js:
            out.append(f"{i['disease_name']} | {i['inference_score']} | {i['inference_gene_symbol']}")
        return out
    
    except Exception:
        return [""]

@mcp.tool
def ctd_chemical_to_go_biological_process(chemical_name: Annotated[str, Field( description="Preferred name of a chemical", min_length=1, max_length=255)]) -> list[str]:
    """
    Given the name of a chemical, return that chemical's associated biological process GO terms from the Comparative Toxicogenomics database (CTD).

    Args:
        chemical_name: A string representing the preferred name of a chemical. This should be a string of at least one character and at most 255 characters.
    Returns:
        A list of strings, where each string is a biological process GO term associated with the given chemical in CTD as well as its target match quantity. If no biological process GO terms are found for the given chemical, an empty list is returned.
    """
    try:
        params = {
            'chemical_name': chemical_name
        }
        encoded_params = urllib.parse.urlencode(params)
        r = requests.get(f"{CHEMBIOTOX_URL}mcp/ctd_chemical_to_go_biological_process?{encoded_params}")
        js = json.loads(r.text)
        
        out = []
        for i in js:
            out.append(f"{i['go_term_name']} | {i['target_match_qty']}")
        return out
    
    except Exception:
        return [""]
    

@mcp.tool
def ctd_chemical_to_go_cellular_component(chemical_name: Annotated[str, Field( description="Preferred name of a chemical", min_length=1, max_length=255)]) -> list[str]:
    """
    Given the name of a chemical, return that chemical's associated cellular component GO terms from the Comparative Toxicogenomics database (CTD).

    Args:
        chemical_name: A string representing the preferred name of a chemical. This should be a string of at least one character and at most 255 characters.
    Returns:
        A list of strings, where each string is a cellular component GO term associated with the given chemical in CTD as well as its target match quantity. If no cellular component GO terms are found for the given chemical, an empty list is returned.
    """
    try:
        params = {
            'chemical_name': chemical_name
        }
        encoded_params = urllib.parse.urlencode(params)
        r = requests.get(f"{CHEMBIOTOX_URL}mcp/ctd_chemical_to_go_cellular_component?{encoded_params}")
        js = json.loads(r.text)
        
        out = []
        for i in js:
            out.append(f"{i['go_term_name']} | {i['target_match_qty']}")
        return out
    
    except Exception:
        return [""]
    
@mcp.tool
def ctd_chemical_to_go_molecular_function(chemical_name: Annotated[str, Field( description="Preferred name of a chemical", min_length=1, max_length=255)]) -> list[str]:
    """
    Given the name of a chemical, return that chemical's associated molecular function GO terms from the Comparative Toxicogenomics database (CTD).

    Args:
        chemical_name: A string representing the preferred name of a chemical. This should be a string of at least one character and at most 255 characters.
    Returns:
        A list of strings, where each string is a molecular function GO term associated with the given chemical in CTD as well as its target match quantity. If no molecular function GO terms are found for the given chemical, an empty list is returned.
    """
    try:
        params = {
            'chemical_name': chemical_name
        }
        encoded_params = urllib.parse.urlencode(params)
        r = requests.get(f"{CHEMBIOTOX_URL}mcp/ctd_chemical_to_go_molecular_function?{encoded_params}")
        js = json.loads(r.text)
        
        out = []
        for i in js:
            out.append(f"{i['go_term_name']} | {i['target_match_qty']}")
        return out
    
    except Exception:
        return [""]
    

@mcp.tool
def tox21_assay_predictions(chemical_name: Annotated[str, Field( description="Preferred name of a chemical", min_length=1, max_length=255)]) -> list[str]:
    """
    Given the name of a chemical, return that chemical's predicted behavior(s) from Tox21 assays. Each item in the output list contains the assay model name and if the chemical was predicted to be active or inactive.

    Args:
        chemical_name: A string representing the preferred name of a chemical. This should be a string of at least one character and at most 255 characters.
    Returns:
        A list of strings, where each string contains the assay model name and if the chemical was predicted to be active or inactive. The assay model name and activity status are formatted as follows: assay_model_name: active/inactive. If no Tox21 predictions are found for the given chemical, an empty list is returned.
    """
    try:
        params = {
            'chemical_name': chemical_name
        }
        encoded_params = urllib.parse.urlencode(params)
        r = requests.get(f"{CHEMBIOTOX_URL}mcp/tox21_assay_predictions?{encoded_params}")
        js = json.loads(r.text)
        
        out = []
        for i in js:
            status = "inactive"
            if i["activity_score"] >= 0.7:
                status = "active"
            out.append(f"{i['assay_model']} | {status}")
        return out
    
    except Exception:
        return [""]
        

@mcp.tool
def drugbank_genes(chemical_name: Annotated[str, Field( description="Preferred name of a chemical", min_length=1, max_length=255)]) -> list[str]:
    """
    Given the name of a chemical, return that chemical's associated gene interactions as documented in DrugBank. This tool returns a list of strings, where each entry is structured like the following: gene_name | interaction

    Args:
        chemical_name: A string representing the preferred name of a chemical. This should be a string of at least one character and at most 255 characters.
    Returns:
        A list of strings, where each string is a gene interaction for the given chemical in DrugBank, structured like the following: gene_name | interaction. If no gene interactions are found for the given chemical, an empty list is returned.
    """
    try:
        params = {
            'chemical_name': chemical_name
        }
        encoded_params = urllib.parse.urlencode(params)
        r = requests.get(f"{CHEMBIOTOX_URL}mcp/drugbank_genes?{encoded_params}")
        js = json.loads(r.text)
        
        out = []
        for i in js:
            out.append(f"{i['genename']} | {i['general_function']}")
        return out
    
    except Exception:
        return [""]

@mcp.tool
def drugbank_atccodes(chemical_name: Annotated[str, Field( description="Preferred name of a chemical", min_length=1, max_length=255)]) -> list[str]:
    """
    Given the name of a chemical, return that chemical's therapeutic properties as documented in DrugBank. Use this tool to retrieve a chemical's organ interactions, therapeutic use, and chemical properties.

    Args:
        chemical_name: A string representing the preferred name of a chemical. This should be a string of at least one character and at most 255 characters.
    Returns:
        A list of strings, where each string is a therapeutic property for the given chemical in DrugBank. If no therapeutic properties are found for the given chemical, an empty list is returned.
    """
    try:
        params = {
            'chemical_name': chemical_name
        }
        encoded_params = urllib.parse.urlencode(params)
        r = requests.get(f"{CHEMBIOTOX_URL}mcp/drugbank_atccodes?{encoded_params}")
        js = json.loads(r.text)
        
        out = []
        for i in js:
            out.append(i['atc_annotation'])
        return out
    
    except Exception:
        return [""]

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

    Args:
        chemical_name: A string representing the preferred name of a chemical. This should be a string of at least one character and at most 255 characters.
    Returns:
        A list of strings, where each string is a GenRA prediction for the given chemical, formatted as follows: category:subcategory - effect. If no GenRA predictions are found for the given chemical, an empty list is returned.

    """
    try:
        params = {
            'chemical_name': chemical_name
        }
        encoded_params = urllib.parse.urlencode(params)
        r = requests.get(f"{CHEMBIOTOX_URL}mcp/genra_results?{encoded_params}")
        js = json.loads(r.text)
        
        out = []
        for i in js:
            out.append(f"{i['genra_category']}:{i['genra_category_name']} - {i['genra_result']}")
        return out
    
    except Exception:
        return [""]


@mcp.tool
def t3db_targets(chemical_name: Annotated[str, Field( description="Preferred name of a chemical", min_length=1, max_length=255)]) -> list[str]:
    """
    Given the name of a chemical, return corresponding targets as documented in the T3DB. This tool can provide information that can help inform how genes and organs interact with a chemical.

    Args:
        chemical_name: A string representing the preferred name of a chemical. This should be a string of at least one character and at most 255 characters.
    Returns:
        A list of strings, where each string is a target associated with the given chemical in T3DB. If no targets are found for the given chemical, an empty list is returned.
    """
    try:
        params = {
            'chemical_name': chemical_name
        }
        encoded_params = urllib.parse.urlencode(params)
        r = requests.get(f"{CHEMBIOTOX_URL}mcp/t3db_targets?{encoded_params}")
        js = json.loads(r.text)
        
        out = []
        for i in js:
            out.append(i['target_name'])
        return out
    
    except Exception:
        return [""]

@mcp.tool
def toxrefdb_cancer_effects(chemical_name: Annotated[str, Field( description="Preferred name of a chemical", min_length=1, max_length=255)]) -> list[str]:
    """
    Given the name of a chemical, return that chemical's toxicological cancer-related effects as reported from studies in ToxRefDB. This tool returns a list of strings, where each item is formatted as follows:

    effect; toxicity type; species; sex; life stage; target

    The toxicity type may be one of the following:
    - ACU - Acute Toxicity
    - CHR - Chronic Toxicity
    - DEV - Developmental Toxicity
    - DNT - Developmental Neurotoxicity
    - MGR - Multigenerational Reproductive Toxicity
    - NEU - Neurological
    - OTH - Other
    - REP - Reproductive Toxicity
    - SAC - Subacute Toxicity
    - SUB - Sub-chronic Toxicity

    Args:
        chemical_name: A string representing the preferred name of a chemical. This should be a string of at least one character and at most 255 characters.
    Returns:
        A list of strings, where each string is a cancer-related effect for the given chemical in ToxRefDB, formatted as follows: effect; toxicity type; species; sex; life stage; target. If no cancer-related effects are found for the given chemical, an empty list is returned.
    """
    try:
        params = {
            'chemical_name': chemical_name
        }
        encoded_params = urllib.parse.urlencode(params)
        r = requests.get(f"{CHEMBIOTOX_URL}mcp/toxrefdb_cancer_effects?{encoded_params}")
        js = json.loads(r.text)
        
        out = []
        for i in js:
            out.append(f"""{i['effect_desc']}; {i['study_type']}; {i['species']}; {i['sex']}; {i['life_stage']}; {i['endpoint_target']}""")
        return out
    
    except Exception:
        return [""]
    
@mcp.tool
def toxrefdb_non_cancer_effects(chemical_name: Annotated[str, Field( description="Preferred name of a chemical", min_length=1, max_length=255)]) -> list[str]:
    """
    Given the name of a chemical, return that chemical's toxicological non-cancer-related effects as reported from studies in ToxRefDB. This tool returns a list of strings, where each item is formatted as follows:

    effect; study type; species; sex; life stage; target

    The study type may be one of the following:
    - ACU - Acute Toxicity
    - CHR - Chronic Toxicity
    - DEV - Developmental Toxicity
    - DNT - Developmental Neurotoxicity
    - MGR - Multigenerational Reproductive Toxicity
    - NEU - Neurological
    - OTH - Other
    - REP - Reproductive Toxicity
    - SAC - Subacute Toxicity
    - SUB - Sub-chronic Toxicity

    Args:
        chemical_name: A string representing the preferred name of a chemical. This should be a string of at least one character and at most 255 characters.
    Returns:
        A list of strings, where each string is a non-cancer-related effect for the given chemical in ToxRefDB, formatted as follows: effect; study type; species; sex; life stage; target. If no non-cancer-related effects are found for the given chemical, an empty list is returned.
    """
    try:
        params = {
            'chemical_name': chemical_name
        }
        encoded_params = urllib.parse.urlencode(params)
        r = requests.get(f"{CHEMBIOTOX_URL}mcp/toxrefdb_non_cancer_effects?{encoded_params}")
        js = json.loads(r.text)
        
        out = []
        for i in js:
            out.append(f"""{i['effect_desc']}; {i['study_type']}; {i['species']}; {i['sex']}; {i['life_stage']}; {i['endpoint_target']}""")
        return out
    
    except Exception:
        return [""]
    
@mcp.tool
def structural_similarity(smiles: Annotated[str, Field( description="SMILES representation of a chemical", min_length=1, max_length=255)], threshold: Annotated[float, Field( description="Tanimoto similarity threshold (default=0.7)")]=0.7) -> list[str]:
    """
    Given a chemical's SMILES representation and a Tanimoto similarity threshold, return structurally similar chemicals with a Tanimoto similarity at or above the specified threshold. Similarity is calculated using Morgan fingerprints.

    Chemicals returned by this tool may be structurally identical to the input chemical (i.e., synonyms of the input chemical).

    Args:
        smiles: A SMILES string representing a chemical's structure. This should be a string of at least one character and at most 255 characters.
        threshold: A float representing the Tanimoto similarity threshold. Unless specified, the threshold is set to a default value of 0.7. Only chemicals with a similarity at or above this threshold will be returned.
    Returns:
        A list of strings representing structurally similar chemicals to the original input smiles, where each string is structured as follows: chemical_name | smiles | tanimoto similarity. The list is ordered from most to least similar chemical, and only chemicals with a Tanimoto similarity at or above the specified threshold are included in the output. If no chemicals with a Tanimoto similarity at or above the specified threshold are found, an empty list is returned. Chemicals returned by this tool may be structurally identical to the input chemical (i.e., synonyms of the input chemical).
    """
    try:
        params = {
            'smiles': smiles,
            'threshold': threshold
        }
        encoded_params = urllib.parse.urlencode(params)
        r = requests.get(f"{CHEMBIOTOX_URL}mcp/structural_similarity?{encoded_params}")
        js = json.loads(r.text)
        
        out = []
        for i in js:
            out.append(f"""{i['preferred_name']} | {i['canonical_smiles']} | {i['similarity']}""")
        return out
    
    except Exception:
        return [""]

@mcp.tool
def structural_similarity_nonidentical(smiles: Annotated[str, Field( description="SMILES representation of a chemical", min_length=1, max_length=255)], threshold: Annotated[float, Field( description="Tanimoto similarity threshold (default=0.7)")]=0.7) -> list[str]:
    """
    Given a chemical's SMILES representation and a Tanimoto similarity threshold, return structurally similar chemicals with a Tanimoto similarity at or above the specified threshold. Similarity is calculated using Morgan fingerprints.

    Chemicals returned by this tool cannot be structurally identical to the input chemical (i.e., no synonyms of the input chemical).
    
    Args:
        smiles: A SMILES string representing a chemical's structure. This should be a string of at least one character and at most 255 characters.
        threshold: A float representing the Tanimoto similarity threshold. Unless specified, the threshold is set to a default value of 0.7. Only chemicals with a similarity at or above this threshold will be returned.
    Returns:
        A list of strings representing structurally similar chemicals to the original input smiles, where each string is structured as follows: chemical_name | smiles | tanimoto similarity. The list is ordered from most to least similar chemical, and only chemicals with a Tanimoto similarity at or above the specified threshold are included in the output. If no chemicals with a Tanimoto similarity at or above the specified threshold are found, an empty list is returned. Chemicals returned by this tool cannot be structurally identical to the input chemical (i.e., no synonyms of the input chemical).
    """
    try:
        params = {
            'smiles': smiles,
            'threshold': threshold
        }
        encoded_params = urllib.parse.urlencode(params)
        r = requests.get(f"{CHEMBIOTOX_URL}mcp/structural_similarity_nonidentical?{encoded_params}")
        js = json.loads(r.text)
        
        out = []
        for i in js:
            out.append(f"""{i['preferred_name']} | {i['canonical_smiles']} | {i['similarity']}""")
        return out
    
    except Exception:
        return [""]

if __name__ == "__main__":
    mcp.run(transport="http", host="127.0.0.1", port=9222)

