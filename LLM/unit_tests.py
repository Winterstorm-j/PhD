import unittest
from unittest.mock import patch
import io # Used to capture print statements

# Import the classes and functions from the script
from agentic import RAG_LLM_Agent, Build_Agent, main

class TestAgentWorkflow(unittest.TestCase):
    """
    Test suite for the agentic AI script.
    """

    def setUp(self):
        """
        Set up common resources for the tests.
        This method is called before each test function.
        """
        self.mock_knowledge_base = {
            "python_docker_template": {
                "latest_python_image": "python:3.11-slim",
                "description": "A standard template for a Python web application."
            }
        }
        self.rag_agent = RAG_LLM_Agent(self.mock_knowledge_base)
        self.build_agent = Build_Agent()

    # --- RAG_LLM_Agent Tests ---

    def test_rag_agent_success(self):
        """
        Test RAG_LLM_Agent's process_query method for a successful case.
        """
        query = "Create a python dockerfile for me."
        expected_params = {
            "base_image": "python:3.11-slim",
            "workdir": "/usr/src/app",
            "dependencies": ["flask", "gunicorn"],
            "expose_port": 5000,
            "start_command": '["gunicorn", "--bind", "0.0.0.0:5000", "app:app"]'
        }
        
        result = self.rag_agent.process_query(query)
        self.assertDictEqual(result, expected_params)

    def test_rag_agent_knowledge_base_miss(self):
        """
        Test that RAG_LLM_Agent raises a ValueError if the knowledge base is missing data.
        """
        empty_kb_agent = RAG_LLM_Agent(knowledge_base={})
        query = "This query will fail."
        
        with self.assertRaises(ValueError) as context:
            empty_kb_agent.process_query(query)
        
        self.assertEqual(str(context.exception), "Could not find relevant information in the knowledge base.")

    # --- Build_Agent Tests ---

    def test_build_agent_execute_build(self):
        """
        Test Build_Agent's execute_build method to ensure it creates the correct Dockerfile.
        """
        build_params = {
            "base_image": "test-python:latest",
            "workdir": "/app",
            "dependencies": ["numpy", "pandas"],
            "expose_port": 8080,
            "start_command": '["python", "main.py"]'
        }
        
        dockerfile = self.build_agent.execute_build(build_params)

        # Check for key components in the generated Dockerfile string
        self.assertIn("FROM test-python:latest", dockerfile)
        self.assertIn("WORKDIR /app", dockerfile)
        self.assertIn("RUN pip install --no-cache-dir numpy pandas", dockerfile)
        self.assertIn("EXPOSE 8080", dockerfile)
        self.assertIn('CMD ["python", "main.py"]', dockerfile)

    # --- Main Function Integration Tests ---
    
    @patch('sys.stdout', new_callable=io.StringIO)
    def test_main_workflow_success(self, mock_stdout):
        """
        Test the main function's successful end-to-end workflow.
        """
        result = main()
        
        # Check the returned result
        self.assertIn("FROM python:3.11-slim", result)
        self.assertIn("CMD [\"gunicorn\", \"--bind\", \"0.0.0.0:5000\", \"app:app\"]", result)
        
        # Check the printed output
        output = mock_stdout.getvalue()
        self.assertIn("✅ Workflow complete! Here is the generated output:", output)

    @patch('agent_script.RAG_LLM_Agent.process_query')
    @patch('sys.stdout', new_callable=io.StringIO)
    def test_main_workflow_failure(self, mock_stdout, mock_process_query):
        """
        Test the main function's error handling by mocking a failure.
        """
        # Configure the mock to raise an exception when called
        error_message = "Simulated LLM failure"
        mock_process_query.side_effect = Exception(error_message)

        result = main()

        # Check that the result is the error message
        self.assertIn(f"💥 An error occurred during the workflow: {error_message}", result)
        
        # Check that the error message was also printed to stdout
        output = mock_stdout.getvalue()
        self.assertIn(f"💥 An error occurred during the workflow: {error_message}", output)


if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)
