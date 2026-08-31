from agents.base_agent import BaseMCPAgent
from schemas.report import ReportFields
from prompts.report_extraction import extract_report_prompt


class ReportAgent(BaseMCPAgent):
    def __init__(self):
        super().__init__("ReportAgent", ReportFields)

    def _get_extraction_prompt(self, content: str) -> str:
        return extract_report_prompt(content)


if __name__ == "__main__":
    agent = ReportAgent()
    agent.run()
