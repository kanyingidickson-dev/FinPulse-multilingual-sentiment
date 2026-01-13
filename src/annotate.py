import click
import json
import os
from datetime import datetime

def load_text_stream(files):
    """Generator that yields records from input files."""
    for file_path in files:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                yield json.loads(line)

@click.command()
@click.option('--input-file', help='Path to raw JSONL file to annotate')
@click.option('--output-file', default='data/annotated_custom.jsonl', help='Path to save annotations')
def annotate(input_file, output_file):
    """
    CLI tool for human-in-the-loop sentiment annotation.
    Prompts user to label text as Positive, Neural, or Negative.
    """
    click.clear()
    click.echo("===========================================")
    click.echo("   FinPulse Sentiment Annotation Tool      ")
    click.echo("   (p: Positive, n: Negative, u: Neutral, s: Skip, q: Quit)")
    click.echo("===========================================")
    
    if not input_file or not os.path.exists(input_file):
        click.echo("Please provide a valid input file.")
        return

    # Load existing to avoid duplicates? (Skipped for simplicity in this version)
    
    mode_map = {'p': 'positive', 'n': 'negative', 'u': 'neutral', 's': 'skip'}
    
    with open(output_file, 'a', encoding='utf-8') as f_out:
        for i, record in enumerate(load_text_stream([input_file])):
            
            # Simple heuristic to skip if already labeled
            if 'label' in record and record['label']:
                continue

            click.echo(f"\n[{record.get('language', 'unknown')}] Text: {record.get('text')}")
            
            while True:
                choice = click.prompt("Label?", type=str).lower()
                
                if choice == 'q':
                    click.echo("Exiting annotation...")
                    return
                
                if choice in mode_map:
                    if choice != 's':
                        record['label'] = mode_map[choice]
                        record['annotated_at'] = datetime.now().isoformat()
                        f_out.write(json.dumps(record) + "\n")
                        f_out.flush() # Ensure it's saved
                    break
                else:
                    click.echo("Invalid input. Use p/n/u/s/q.")

if __name__ == '__main__':
    annotate()
