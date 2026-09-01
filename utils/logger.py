import logging
import json
from datetime import datetime
from colorama import Fore, Style, init

init(autoreset=True)

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
        print(f"\n{Fore.CYAN}💭 THOUGHT: {Style.RESET_ALL}{thought}")
        
    def log_action(self, action: str, tool: str):
        """Log agent's action"""
        print(f"{Fore.YELLOW}🔧 ACTION: {Style.RESET_ALL}Using tool '{tool}' - {action}")
        
    def log_observation(self, observation: str):
        """Log tool results"""
        preview = observation[:200] + "..." if len(observation) > 200 else observation
        print(f"{Fore.GREEN}👁️  OBSERVATION: {Style.RESET_ALL}{preview}")
        
    def log_response(self, response: str):
        """Log final response"""
        print(f"\n{Fore.MAGENTA}✨ FINAL RESPONSE:{Style.RESET_ALL}\n{response}\n")
        
    def log_error(self, error: str):
        """Log errors"""
        print(f"{Fore.RED}❌ ERROR: {Style.RESET_ALL}{error}")
        
    def log_iteration(self, iteration: int, max_iterations: int):
        """Log iteration count"""
        print(f"\n{Fore.BLUE}{'='*60}")
        print(f"🔄 ITERATION {iteration}/{max_iterations}")
        print(f"{'='*60}{Style.RESET_ALL}")

logger = AgentLogger()