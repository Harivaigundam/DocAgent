from agents.invoice_agent import InvoiceAgent
from agents.receipt_agent import ReceiptAgent
from agents.contract_agent import ContractAgent
from agents.report_agent import ReportAgent
from agents.guardrail_agent import mcp as guardrail_mcp
from agents.parser_agent import mcp as parser_mcp
from agents.validator_agent import mcp as validator_mcp
from agents.summarizer_agent import mcp as summarizer_mcp

AGENT_MAP = {
    "invoice": InvoiceAgent,
    "receipt": ReceiptAgent,
    "contract": ContractAgent,
    "report": ReportAgent,
}

SUBTASK_MCP_MAP = {
    "guardrail": guardrail_mcp,
    "parser": parser_mcp,
    "validator": validator_mcp,
    "summarizer": summarizer_mcp,
}
