from agents.base_agent import BaseMCPAgent
from schemas.receipt import ReceiptFields
from prompts.receipt_extraction import extract_receipt_prompt


class ReceiptAgent(BaseMCPAgent):
    def __init__(self):
        super().__init__("ReceiptAgent", ReceiptFields)

    def _get_extraction_prompt(self, content: str) -> str:
        return extract_receipt_prompt(content)


if __name__ == "__main__":
    agent = ReceiptAgent()
    agent.run()
