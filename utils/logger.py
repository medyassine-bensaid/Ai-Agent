import logging
import sys

from colorama import Fore, Style, init

init(autoreset=True)


def safe_print(text: str):
    """print() that never raises: consoles/pipes that cannot encode emoji
    (e.g. cp1252 on Windows) get replacement characters instead of crashing
    the agent run."""
    try:
        print(text)
    except UnicodeEncodeError:
        enc = getattr(sys.stdout, "encoding", None) or "ascii"
        print(text.encode(enc, errors="replace").decode(enc, errors="replace"))


class AgentLogger:
    """Structured logger for agent operations"""

    def __init__(self, name="DevOpsAgent"):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.INFO)

        # Console handler with color
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        self.logger.addHandler(console_handler)

    def log_thought(self, thought: str):
        """Log agent's reasoning"""
        safe_print(f"\n{Fore.CYAN}💭 THOUGHT: {Style.RESET_ALL}{thought}")

    def log_action(self, action: str, tool: str):
        """Log agent's action"""
        safe_print(
            f"{Fore.YELLOW}🔧 ACTION: {Style.RESET_ALL}Using tool '{tool}' - {action}"
        )

    def log_observation(self, observation: str):
        """Log tool results"""
        preview = observation[:200] + "..." if len(observation) > 200 else observation
        safe_print(f"{Fore.GREEN}👁️  OBSERVATION: {Style.RESET_ALL}{preview}")

    def log_response(self, response: str):
        """Log final response"""
        safe_print(f"\n{Fore.MAGENTA}✨ FINAL RESPONSE:{Style.RESET_ALL}\n{response}\n")

    def log_error(self, error: str):
        """Log errors"""
        safe_print(f"{Fore.RED}❌ ERROR: {Style.RESET_ALL}{error}")

    def log_iteration(self, iteration: int, max_iterations: int):
        """Log iteration count"""
        safe_print(f"\n{Fore.BLUE}{'='*60}")
        safe_print(f"🔄 ITERATION {iteration}/{max_iterations}")
        safe_print(f"{'='*60}{Style.RESET_ALL}")


logger = AgentLogger()
