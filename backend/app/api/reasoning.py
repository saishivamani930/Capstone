from fastapi import APIRouter, HTTPException
from app.schemas.clinical import ClinicalEntitiesResponse
from app.schemas.reasoning import ReasoningResponse, CandidateDisease
from app.knowledge_graph.client import Neo4jClient
from app.reasoning.wikidata_client import WikidataClient
from app.reasoning.missing_info import MissingInfoEngine
from app.reasoning.risk_analyzer import RiskAnalyzer

router = APIRouter(prefix="/api/reasoning", tags=["Knowledge Graph & Reasoning"])

neo4j_client = Neo4jClient()
wiki_client = WikidataClient()
missing_engine = MissingInfoEngine(wiki_client)
risk_analyzer = RiskAnalyzer()

@router.post("/process", response_model=ReasoningResponse)
async def process_clinical_reasoning(nlp_output: ClinicalEntitiesResponse, patient_id: str = "patient_001"):
    try:
        # Step 1: Extract entities and resolve Wikidata Q-IDs for symptoms
        entities_dict = []
        symptom_qids = []
        present_symptom_names = []

        for entity in nlp_output.entities:
            e_dict = entity.model_dump() if hasattr(entity, "model_dump") else entity.dict()
            
            # If entity is a symptom, resolve its Wikidata Q-ID
            if e_dict.get("category") == "symptom":
                qid = await wiki_client.get_qid_for_entity(e_dict["text"])
                e_dict["wikidata_id"] = qid
                if e_dict.get("status") == "present":
                    present_symptom_names.append(e_dict["text"])
                    if qid:
                        symptom_qids.append(qid)
            
            entities_dict.append(e_dict)

        relations_dict = [
            rel.model_dump() if hasattr(rel, "model_dump") else rel.dict() 
            for rel in nlp_output.relations
        ]

        # Step 2: Sync to Neo4j Knowledge Graph
        neo4j_client.sync_patient_nlp(patient_id, entities_dict, relations_dict)

        # Step 3: Run Wikidata SPARQL Differential Diagnoses
        raw_candidates = await wiki_client.find_candidate_diseases(symptom_qids)
        
        candidate_diseases = [
            CandidateDisease(
                wikidata_id=c["wikidata_id"],
                disease_name=c["disease_name"],
                matched_symptoms=present_symptom_names,
                match_count=c["match_count"]
            )
            for c in raw_candidates
        ]

        # Step 4: Detect Missing Information & Generate Questions
        missing_info = await missing_engine.analyze(present_symptom_names, candidate_diseases)

        # Step 5: Construct graph visualizer output format for Member 4 UI
        nodes = [
            {
                "id": e["id"],
                "label": e["text"],
                "category": e["category"],
                "status": e.get("status", "present")
            }
            for e in entities_dict
        ]
        
        edges = [
            {
                "source": r["source_id"],
                "target": r["target_id"],
                "type": r["relation_type"]
            }
            for r in relations_dict
        ]

        # Step 6: Perform Cardiology Risk Analysis
        risk_res = risk_analyzer.analyze_risk(present_symptom_names, candidate_diseases)

        return ReasoningResponse(
            patient_id=patient_id,
            graph_nodes=nodes,
            graph_relationships=edges,
            candidate_diseases=candidate_diseases,
            missing_information=missing_info,
            risk_analysis=risk_res
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Reasoning processing failed: {str(e)}")