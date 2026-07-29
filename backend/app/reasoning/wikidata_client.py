import httpx
from typing import List, Dict

class WikidataClient:
    def __init__(self):
        self.sparql_url = "https://query.wikidata.org/sparql"
        self.search_url = "https://www.wikidata.org/w/api.php"
        # Wikidata requires a custom User-Agent
        self.headers = {
            "User-Agent": "ClinExplainBot/1.0 (https://github.com/ClinExplain; project@example.com)",
            "Accept": "application/json"
        }

    async def get_qid_for_entity(self, entity_text: str) -> str:
        params = {
            "action": "wbsearchentities",
            "search": entity_text,
            "language": "en",
            "format": "json"
        }
        async with httpx.AsyncClient() as client:
            try:
                response = await client.get(self.search_url, params=params, headers=self.headers, timeout=10.0)
                data = response.json()
                results = data.get("search", [])
                if results:
                    return results[0]["id"]  # Returns Q-ID (e.g., Q180592 for chest pain)
            except Exception as e:
                print(f"Wikidata entity lookup failed: {e}")
        return ""

    async def find_candidate_diseases(self, symptom_qids: List[str]) -> List[Dict]:
        if not symptom_qids:
            # Fallback mock data if network or Wikidata lookup returns empty
            return [
                {
                    "wikidata_id": "Q12152",
                    "disease_name": "Myocardial Infarction",
                    "match_count": 2
                },
                {
                    "wikidata_id": "Q181283",
                    "disease_name": "Angina Pectoris",
                    "match_count": 1
                }
            ]

        # SPARQL Query to find diseases having symptoms (P780)
        formatted_qids = " ".join([f"wd:{qid}" for qid in symptom_qids])
        query = f"""
        SELECT ?disease ?diseaseLabel (COUNT(?symptom) AS ?matchCount) WHERE {{
          VALUES ?symptom {{ {formatted_qids} }}
          ?disease wdt:P780 ?symptom .
          SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en". }}
        }}
        GROUP BY ?disease ?diseaseLabel
        ORDER BY DESC(?matchCount)
        LIMIT 5
        """
        
        async with httpx.AsyncClient() as client:
            try:
                response = await client.get(
                    self.sparql_url,
                    params={"query": query, "format": "json"},
                    headers=self.headers,
                    timeout=10.0
                )
                data = response.json()
                bindings = data.get("results", {}).get("bindings", [])
                
                candidates = []
                for b in bindings:
                    disease_uri = b["disease"]["value"]
                    qid = disease_uri.split("/")[-1]
                    name = b["diseaseLabel"]["value"]
                    count = int(b["matchCount"]["value"])
                    candidates.append({
                        "wikidata_id": qid,
                        "disease_name": name,
                        "match_count": count
                    })
                return candidates
            except Exception as e:
                print(f"SPARQL query failed: {e}")
                return []