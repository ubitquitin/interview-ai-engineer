import asyncio
import json
import logging
import hashlib
from typing import Optional
from tqdm.asyncio import tqdm

from langchain_ollama import ChatOllama
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document

from src.models import WarningLetterDocument, WarningLetterMetadata

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


@staticmethod
def inline_refs(schema: dict) -> dict:
    """Helper function to resolve all $ref/$defs into a flat schema."""
    defs = schema.pop("$defs", {})

    def resolve(obj):
        if isinstance(obj, dict):
            if "$ref" in obj:
                ref_name = obj["$ref"].split("/")[-1]
                return resolve(defs[ref_name])
            return {k: resolve(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [resolve(i) for i in obj]
        return obj

    return resolve(schema)


def pre_filter_pharma(input_file, output_file):
    # The official "Big Three" Life Science Centers
    allowed_offices = {
        "Center for Drug Evaluation and Research (CDER)",
        "Center for Devices and Radiological Health",
        "Center for Biologics Evaluation and Research (CBER)"
    }

    count = 0
    total = 0

    with open(input_file, "r") as f_in, open(output_file, "w") as f_out:
        for line in f_in:
            total += 1
            record = json.loads(line)
            office = record.get("metadata", {}).get("issuing_office", "").strip()

            # Check if the office is in our allowed set
            if office in allowed_offices:
                f_out.write(json.dumps(record) + "\n")
                count += 1

    print(f"Processed {total} total records.")
    print(f"Kept {count} records from CDER, CDRH, and CBER for schematization.")


class LocalSchematizer:
    def __init__(self, model_name: str = "llama3"):
        # 1. Initialize the base LLM
        self.llm = ChatOllama(
            model=model_name,
            temperature=0,
            num_ctx=4096
        )

        schema_dict = WarningLetterDocument.model_json_schema()
        flat_schema = inline_refs(schema_dict)

        # Remove injected fields from the schema
        for field in ("metadata", "content_hash"):
            flat_schema["properties"].pop(field, None)
            if field in flat_schema.get("required", []):
                flat_schema["required"].remove(field)

        self.structured_llm = self.llm.with_structured_output(flat_schema, method="json_schema")

    async def process_letter(self, raw_content: str, metadata_dict: dict) -> Optional[WarningLetterDocument]:
        """Extracts structured data directly into a Pydantic object."""
        try:
            prompt = (
                "You are an FDA Specialist. Extract the following Warning Letter text into "
                "structured JSON format. Focus on the 'deficiencies' and 'cfr_reference' fields.\n\n"
                f"LETTER TEXT:\n{raw_content[:12000]}"
            )

            # This will now return a real WarningLetterDocument instance
            raw_json = await self.structured_llm.ainvoke(prompt)

            # Inject the fields the LLM doesn't produce
            raw_json["metadata"] = metadata_dict
            raw_json["content_hash"] = hashlib.md5(raw_content.encode()).hexdigest()

            return WarningLetterDocument(**raw_json)

        except Exception as e:
            # If it still fails, it's often a context window or specific character issue
            logger.error(f"Structured extraction failed: {e}")
            return None


async def run_schematization(input_file, output_file):
    schematizer = LocalSchematizer()
    semaphore = asyncio.BoundedSemaphore(4)

    async def sem_process(record):
        async with semaphore:
            return await schematizer.process_letter(record["content"], record["metadata"])

    logger.info("Starting parallelized schematization...")

    records = []
    with open(input_file, "r") as f_in:
        for line in f_in:
            records.append(json.loads(line))

    # Create tasks
    tasks = [sem_process(r) for r in records]

    # Open file in APPEND mode for safety
    with open(output_file, "a") as f_out:
        # tqdm gives you: [Remaining Time, Iterations/Second, Progress Bar]
        for coro in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="Processing"):
            try:
                gold_doc = await coro
                if gold_doc:
                    f_out.write(gold_doc.model_dump_json() + "\n")
                    f_out.flush() # Force write to disk immediately
            except Exception as e:
                logger.error(f"Task failed: {e}")

    logger.info(f"Schematization complete. Results saved to {output_file}")


def clean_dupes(input_path, output_path):
    seen_content_hashes = set()
    seen_urls = set()
    duplicate_urls = []
    duplicate_deficiencies = 0
    kept = 0

    with open(input_path) as f_in, open(output_path, "w") as f_out:
        for line in f_in:
            data = json.loads(line)
            url = data["metadata"]["url"]

            # Deduplicate at the letter level first
            if url in seen_urls:
                duplicate_urls.append(url)
                continue
            seen_urls.add(url)

            # Deduplicate at the deficiency level within each letter
            clean_deficiencies = []
            for deficiency in data.get("deficiencies", []):
                content = (
                    f"Regulation: {deficiency['cfr_reference']}\n"
                    f"Violation: {deficiency['title']}\n"
                    f"Description: {deficiency['description']}\n"
                    f"Evidence: {deficiency['evidence']}"
                )
                h = hashlib.md5(content.encode()).hexdigest()
                if h in seen_content_hashes:
                    duplicate_deficiencies += 1
                    continue
                seen_content_hashes.add(h)
                clean_deficiencies.append(deficiency)

            data["deficiencies"] = clean_deficiencies
            f_out.write(json.dumps(data) + "\n")
            kept += 1

    print(f"Duplicate letters (by URL): {len(duplicate_urls)}")
    print(f"Duplicate deficiencies (by content): {duplicate_deficiencies}")
    print(f"Clean letters written: {kept}")


def hydrate_vector_db(input_path: str):
    logger.info("Initializing Atomic Deficiency Hydration...")
    # FastEmbed uses ONNX Runtime - much lighter than PyTorch
    embeddings = FastEmbedEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    docs = []
    ids = []

    with open(input_path, "r") as f:
        for line in f:
            data = json.loads(line)
            metadata = data['metadata']

            # Atomic Chunking: Iterate through each deficiency individually
            for i, deficiency in enumerate(data.get('deficiencies', [])):
                # Construct an atomic context focused on the specific violation
                content = f"""
                    CFR: {deficiency['cfr_reference']}
                    Violation: {deficiency['title']}

                    Description:
                    {deficiency['description']}

                    Evidence:
                    {deficiency['evidence']}

                    Required Action:
                    {deficiency['required_action']}
                    """

                # We keep the company metadata so the Agent can cite its source
                doc_metadata = {
                    **metadata,
                    "cfr_reference": deficiency['cfr_reference'],
                    "violation_title": deficiency['title']
                }

                docs.append(Document(page_content=content, metadata=doc_metadata))

                # Unique ID: Hash letter content
                unique_id = hashlib.md5(content.encode()).hexdigest()

                ids.append(unique_id)

    vector_db = Chroma.from_documents(
        documents=docs,
        ids=ids,
        embedding=embeddings,
        persist_directory="data/vector_db"
    )

    logger.info(f"Hydrated {len(docs)} atomic deficiencies into the Vector DB.")


if __name__ == "__main__":
    input_file = "data/warning_letters_raw.jsonl"
    interim_file = "data/warning_letters_filtered.jsonl"
    output_file = "data/warning_letters_schematized.jsonl"
    final_file = "data/warning_letters_final.jsonl"

    # pre_filter_pharma(input_file, interim_file)

    # asyncio.run(run_schematization(interim_file, output_file))

    clean_dupes(output_file, final_file)

    hydrate_vector_db(final_file)

