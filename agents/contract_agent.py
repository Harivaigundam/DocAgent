from agents.base_agent import BaseMCPAgent
from schemas.contract import ContractFields
from prompts.contract_extraction import extract_contract_prompt


class ContractAgent(BaseMCPAgent):
    def __init__(self):
        super().__init__("ContractAgent", ContractFields)

    def _get_extraction_prompt(self, content: str) -> str:
        return extract_contract_prompt(content)


if __name__ == "__main__":
    agent = ContractAgent()
    agent.run()
