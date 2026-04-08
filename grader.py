import sys
import argparse
import json
from server.ticket_router_environment import (
    BasicRoutingRubric,
    ExtractionRoutingRubric,
    PIIRedactionRubric,
    TicketObservation
)

def main():
    parser = argparse.ArgumentParser(description="OpenEnv Grader for Ticket Router")
    parser.add_argument("--task", type=str, required=True, help="Task ID to grade")
    parser.add_argument("--submission", type=str, help="Path to submission file (optional)")
    parser.add_argument("--metadata", type=str, help="JSON string of trajectory metadata")

    args = parser.parse_args()

    # Map task names to rubrics
    rubrics = {
        "basic_routing": BasicRoutingRubric(),
        "extraction_routing": ExtractionRoutingRubric(),
        "pii_redaction": PIIRedactionRubric(),
    }

    if args.task not in rubrics:
        print(f"Error: Unknown task {args.task}")
        sys.exit(1)

    rubric = rubrics[args.task]
    
    # In a real validation scenario, the validator passes the trajectory metadata
    # as a JSON string. We parse it to recreate the observation needed by our Rubric.
    metadata = {}
    if args.metadata:
        try:
            metadata = json.loads(args.metadata)
        except json.JSONDecodeError:
            pass

    # Create a dummy observation that carries the metadata needed for grading
    # The Rubrics we implemented expect:
    # metadata['current_ticket'], metadata['submitted_department'], etc.
    obs = TicketObservation(
        done=True,
        metadata=metadata
    )

    # Compute score
    score = rubric(None, obs)
    
    # Meta validator expects the score to be printed or returned in a specific JSON format
    result = {
        "score": score,
        "task": args.task,
        "message": f"Successfully graded {args.task}"
    }
    print(json.dumps(result))

if __name__ == "__main__":
    main()
