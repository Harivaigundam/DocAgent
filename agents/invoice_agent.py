from agents.base_agent import BaseMCPAgent
from schemas.invoice import InvoiceFields
from prompts.invoice_extraction import extract_invoice_prompt


class InvoiceAgent(BaseMCPAgent):
    def __init__(self):
        super().__init__("InvoiceAgent", InvoiceFields)

    def _get_extraction_prompt(self, content: str) -> str:
        return extract_invoice_prompt(content)


if __name__ == "__main__":
    agent = InvoiceAgent()
    agent.run()
