"""
Command-line interface for the traffic violation detection pipeline.
"""

import argparse
import sys
from pathlib import Path


def main():
    """Main entry point for the CLI."""
    parser = argparse.ArgumentParser(
        description="Traffic Violation Detection Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process a single image
  python -m traffic_violation_detector.main --image path/to/image.jpg

  # Process all images in a directory
  python -m traffic_violation_detector.main --directory path/to/images/

  # Process a video file
  python -m traffic_violation_detector.main --video path/to/video.mp4

  # Process video with preview window
  python -m traffic_violation_detector.main --video path/to/video.mp4 --preview

  # Use custom config file
  python -m traffic_violation_detector.main --image image.jpg --config custom_config.yaml
        """,
    )
    
    # Input options (mutually exclusive)
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--image", "-i",
        type=str,
        help="Path to a single image file to process",
    )
    input_group.add_argument(
        "--directory", "-d",
        type=str,
        help="Path to a directory containing images to process",
    )
    input_group.add_argument(
        "--video", "-v",
        type=str,
        help="Path to a video file to process",
    )
    
    # Configuration options
    parser.add_argument(
        "--config", "-c",
        type=str,
        default=None,
        help="Path to custom configuration YAML file",
    )
    
    # Video processing options
    parser.add_argument(
        "--frame-skip",
        type=int,
        default=5,
        help="Process every Nth frame in video (default: 5)",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Maximum number of frames to process from video",
    )
    parser.add_argument(
        "--preview",
        action="store_true",
        help="Show preview window during video processing",
    )
    
    # Output options
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Override output directory from config",
    )
    parser.add_argument(
        "--no-save-images",
        action="store_true",
        help="Don't save annotated violation images",
    )
    
    # Logging options
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging (DEBUG level)",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress all output except errors",
    )
    
    args = parser.parse_args()
    
    # Import here to avoid slow startup for --help
    from .config import load_config
    from .pipeline import TrafficViolationPipeline
    
    # Load configuration
    config_path = Path(args.config) if args.config else None
    config = load_config(config_path)
    
    # Apply CLI overrides
    if args.verbose:
        config.logging.level = "DEBUG"
    elif args.quiet:
        config.logging.level = "ERROR"
    
    if args.output_dir:
        config.output.output_dir = args.output_dir
    
    if args.no_save_images:
        config.output.save_images = False
    
    # Initialize pipeline
    pipeline = TrafficViolationPipeline(config=config)
    
    try:
        if args.image:
            # Process single image
            result = pipeline.process_image(args.image)
            
            print(f"\n{'='*50}")
            print(f"Processing complete: {args.image}")
            print(f"Two-wheelers detected: {result.two_wheelers_detected}")
            print(f"Violations found: {result.violations_found}")
            print(f"Processing time: {result.processing_time_ms:.2f}ms")
            
            if result.violations:
                print(f"\nViolations:")
                for v in result.violations:
                    print(f"  - {v.violation_id}: {v.license_plate or 'PLATE NOT READABLE'}")
            
            print(f"\nResults saved to: {config.get_violations_csv_path()}")
            print(f"{'='*50}")
        
        elif args.directory:
            # Process directory
            results = pipeline.process_directory(args.directory)
            
            total_vehicles = sum(r.two_wheelers_detected for r in results)
            total_violations = sum(r.violations_found for r in results)
            
            print(f"\n{'='*50}")
            print(f"Directory processing complete: {args.directory}")
            print(f"Images processed: {len(results)}")
            print(f"Total two-wheelers detected: {total_vehicles}")
            print(f"Total violations found: {total_violations}")
            print(f"\nResults saved to: {config.get_violations_csv_path()}")
            print(f"{'='*50}")
        
        elif args.video:
            # Process video
            results = pipeline.process_video(
                args.video,
                frame_skip=args.frame_skip,
                max_frames=args.max_frames,
                show_preview=args.preview,
            )
            
            total_vehicles = sum(r.two_wheelers_detected for r in results)
            total_violations = sum(r.violations_found for r in results)
            
            print(f"\n{'='*50}")
            print(f"Video processing complete: {args.video}")
            print(f"Frames processed: {len(results)}")
            print(f"Total two-wheelers detected: {total_vehicles}")
            print(f"Total violations found: {total_violations}")
            print(f"\nResults saved to: {config.get_violations_csv_path()}")
            print(f"{'='*50}")
        
        return 0
    
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
