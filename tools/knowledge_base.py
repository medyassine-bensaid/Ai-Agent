from langchain_core.tools import tool
from utils.logger import logger

# Simulated knowledge base (in production, this would be a vector database)
DEVOPS_KNOWLEDGE = {
    "ci/cd": """
    CI/CD (Continuous Integration/Continuous Deployment) is a DevOps practice that automates the software delivery process.
    
    Continuous Integration (CI): Developers frequently merge code changes into a central repository, where automated builds and tests run.
    
    Continuous Deployment (CD): Automatically deploys all code changes that pass automated tests to production.
    
    Key benefits:
    - Faster time to market
    - Reduced manual errors
    - Immediate feedback
    - Better collaboration
    
    Common tools: Jenkins, GitLab CI, GitHub Actions, CircleCI, Travis CI
    """,
    
    "docker": """
    Docker is a containerization platform that packages applications and dependencies into containers.
    
    Key concepts:
    - Containers: Lightweight, standalone executable packages
    - Images: Templates for creating containers
    - Dockerfile: Instructions for building images
    - Docker Compose: Tool for defining multi-container applications
    
    Benefits:
    - Consistency across environments
    - Isolation and security
    - Resource efficiency
    - Rapid deployment
    """,
    
    "kubernetes": """
    Kubernetes (K8s) is a container orchestration platform for automating deployment, scaling, and management of containerized applications.
    
    Core concepts:
    - Pods: Smallest deployable units
    - Services: Expose applications
    - Deployments: Manage pod lifecycles
    - Namespaces: Organize resources
    
    Benefits:
    - Auto-scaling
    - Self-healing
    - Load balancing
    - Rolling updates
    """,
    
    "infrastructure as code": """
    Infrastructure as Code (IaC) is managing infrastructure through code rather than manual processes.
    
    Key principles:
    - Version control for infrastructure
    - Reproducible environments
    - Automated provisioning
    - Documentation through code
    
    Common tools:
    - Terraform: Cloud-agnostic provisioning
    - CloudFormation: AWS-specific
    - Ansible: Configuration management
    - Pulumi: Programming language-based IaC
    """,
    
    "devops culture": """
    DevOps is a cultural philosophy that emphasizes collaboration between development and operations teams.
    
    Core principles (CALMS):
    - Culture: Shared responsibility
    - Automation: Eliminate manual work
    - Lean: Continuous improvement
    - Measurement: Data-driven decisions
    - Sharing: Knowledge and feedback
    
    The Three Ways:
    1. Flow: Optimize delivery pipeline
    2. Feedback: Amplify feedback loops
    3. Continuous Learning: Experimentation and learning
    """
}

@tool
def knowledge_base_tool(topic: str) -> str:
    """
    Query the DevOps knowledge base for established concepts and best practices.
    Use this for foundational DevOps knowledge that doesn't change frequently.
    
    Args:
        topic: The DevOps topic to query (e.g., "ci/cd", "docker", "kubernetes")
    
    Returns:
        Knowledge base information about the topic
    """
    try:
        logger.log_action(f"Querying knowledge base for: '{topic}'", "knowledge_base")
        
        # Simple keyword matching (in production, use vector similarity)
        topic_lower = topic.lower()
        
        # Find matching topics
        matches = []
        for key, value in DEVOPS_KNOWLEDGE.items():
            if any(word in key for word in topic_lower.split()) or key in topic_lower:
                matches.append(f"=== {key.upper()} ===\n{value}")
        
        if matches:
            result = "\n\n".join(matches)
            logger.log_observation(f"Found {len(matches)} knowledge base entries")
            return result
        else:
            logger.log_observation("No exact match in knowledge base")
            return f"No specific knowledge found for '{topic}'. Consider using web_search for this query."
            
    except Exception as e:
        error_msg = f"Knowledge base query failed: {str(e)}"
        logger.log_error(error_msg)
        return error_msg