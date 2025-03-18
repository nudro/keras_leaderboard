#!/usr/bin/env python
"""
Command-line interface for Keras Leaderboard.
"""

import os
import sys
import argparse
from keras_leaderboard.leaderboard import KerasLeaderboard


def main():
    """Entry point for the keras-leaderboard command-line tool."""
    parser = argparse.ArgumentParser(
        description="Keras Leaderboard - Track and compare Keras models"
    )
    
    # Main commands
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Create command
    create_parser = subparsers.add_parser("create", help="Create a new leaderboard")
    create_parser.add_argument(
        "--name", 
        type=str, 
        default="keras_leaderboard", 
        help="Name of the leaderboard"
    )
    create_parser.add_argument(
        "--output-dir", 
        type=str, 
        default="./leaderboard_output", 
        help="Directory to save leaderboard outputs"
    )
    
    # Run command
    run_parser = subparsers.add_parser("run", help="Run models and add to leaderboard")
    run_parser.add_argument(
        "--model", 
        type=str, 
        choices=["basic_cnn", "basic_vgg", "custom"], 
        required=True, 
        help="Model type to run"
    )
    run_parser.add_argument(
        "--data-dir", 
        type=str, 
        default="./data", 
        help="Directory containing training data"
    )
    run_parser.add_argument(
        "--epochs", 
        type=int, 
        default=10, 
        help="Number of epochs to train"
    )
    run_parser.add_argument(
        "--batch-size", 
        type=int, 
        default=32, 
        help="Batch size for training"
    )
    run_parser.add_argument(
        "--leaderboard-path", 
        type=str, 
        default="./leaderboard_output", 
        help="Path to existing leaderboard"
    )
    
    # Display command
    display_parser = subparsers.add_parser("display", help="Display the leaderboard")
    display_parser.add_argument(
        "--path", 
        type=str, 
        default="./leaderboard_output", 
        help="Path to the leaderboard directory"
    )
    display_parser.add_argument(
        "--sort-by", 
        type=str, 
        default="val_accuracy", 
        help="Metric to sort the leaderboard by"
    )
    display_parser.add_argument(
        "--export", 
        type=str, 
        help="Export the leaderboard to a CSV file"
    )
    
    # Parse arguments
    args = parser.parse_args()
    
    # If no command is specified, show help
    if args.command is None:
        parser.print_help()
        sys.exit(1)
    
    # Handle create command
    if args.command == "create":
        os.makedirs(args.output_dir, exist_ok=True)
        leaderboard = KerasLeaderboard(output_dir=args.output_dir)
        print(f"Created new leaderboard at {args.output_dir}")
    
    # Handle run command
    elif args.command == "run":
        try:
            leaderboard = KerasLeaderboard(output_dir=args.leaderboard_path)
            
            if args.model == "basic_cnn":
                # Import and run basic CNN model
                from keras_leaderboard.models.basic_cnn import build_and_train_basic_cnn
                model, history = build_and_train_basic_cnn(
                    data_dir=args.data_dir,
                    epochs=args.epochs,
                    batch_size=args.batch_size
                )
                leaderboard.add_model(model, "basic_cnn", history=history)
                
            elif args.model == "basic_vgg":
                # Import and run VGG model
                from keras_leaderboard.models.basic_vgg import build_and_train_basic_vgg
                model, history = build_and_train_basic_vgg(
                    data_dir=args.data_dir,
                    epochs=args.epochs,
                    batch_size=args.batch_size
                )
                leaderboard.add_model(model, "basic_vgg", history=history)
                
            else:
                print("Custom model loading not yet implemented in CLI.")
                print("Please use the Python API for custom models.")
                sys.exit(1)
                
            print(f"Successfully added {args.model} to the leaderboard")
                
        except Exception as e:
            print(f"Error running model: {e}")
            sys.exit(1)
    
    # Handle display command
    elif args.command == "display":
        try:
            leaderboard = KerasLeaderboard(output_dir=args.path)
            leaderboard.display(sort_by=args.sort_by)
            
            if args.export:
                leaderboard.export_csv(args.export)
                print(f"Exported leaderboard to {args.export}")
                
        except Exception as e:
            print(f"Error displaying leaderboard: {e}")
            sys.exit(1)


if __name__ == "__main__":
    main() 