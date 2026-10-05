import time
import os
import ollama
from telemetry_parser import TelemetryParser


class RaceEngineer:

  def __init__(self, outputs_dir="outputs", poll_interval=5):
    self.parser = TelemetryParser(outputs_dir=outputs_dir)
    self.poll_interval = poll_interval
    self.last_processed_file = None

  def format_prompt(self, stats, global_stats):
    """Constructs a professional race engineer prompt using session data."""
    prompt = f"""
    You are an expert AI race engineer communicating over the pit radio during an Assetto Corsa reinforcement learning training session.
    
    Here is the global training context so far:
    - Total attempts/laps driven: {global_stats['total_attempts']}
    - Total cumulative steps: {global_stats['cumulative_steps']}
    - Estimated active driving time: {global_stats['estimated_time_min']} minutes
    
    Here is the telemetry summary for the most recent completed run:
    - File: {stats['file_path']}
    - Episode Length: {stats['total_steps']} steps
    - Peak Speed: {stats['max_speed_ms']:.2f} m/s
    - Mean Speed: {stats['mean_speed_ms']:.2f} m/s
    - Max Track Progress: {stats['max_track_progress']:.1f}%
    - Max Lateral G-Force: {stats['max_lateral_g']:.2f} G
    - Mean Rear Tire Slip Ratio: {stats['mean_rear_slip']:.3f}
    - Went Off Track: {stats['went_off_track']}
    
    Write a concise, plain-English 2-3 sentence race debrief evaluating the AI's performance, how far it made it around the track, and any potential issues (like tire slip or going off track). Keep it grounded and technical like a real pit wall engineer.
    """
    return prompt

  def generate_debrief(self, stats, global_stats):
    """Sends the formatted prompt to Llama 3.1 via Ollama."""
    prompt = self.format_prompt(stats, global_stats)
    try:
      response = ollama.chat(
          model="llama3.1",
          messages=[{
              "role": "system",
              "content": (
                  "You are a professional motorsport race engineer. Give direct,"
                  " technical debriefs based strictly on telemetry data."
              ),
          }, {
              "role": "user",
              "content": prompt,
          }],
      )
      return response["message"]["content"]
    except Exception as e:
      return f"Error communicating with Ollama: {e}"

  def start_listener(self):
    """Runs a continuous background loop to detect new parquet files and debrief them."""
    print("Race Engineer listener active. Waiting for new training runs...")
    
    # Initialize past files so it only triggers on NEW runs generated from now on
    try:
      self.last_processed_file = self.parser.get_latest_parquet_file()
      print(f"Current latest baseline set to: {self.last_processed_file}")
    except FileNotFoundError:
      print("No existing parquet files found yet. Waiting for first run...")
      self.last_processed_file = None

    while True:
      try:
        latest_file = self.parser.get_latest_parquet_file()
        
        # If a new file appears that we haven't processed yet
        if latest_file != self.last_processed_file:
          print(f"\n[TRIGGER] New training run detected: {latest_file}")
          self.last_processed_file = latest_file
          
          # Small delay to ensure file write is fully finalized
          time.sleep(1)

          # Parse data and call Ollama
          stats, _ = self.parser.parse_latest_run()
          global_stats = self.parser.get_global_training_stats()

          print("Analyzing telemetry and generating Ollama debrief...")
          debrief = self.generate_debrief(stats, global_stats)

          print("\n--- RACE ENGINEER DEBRIEF ---")
          print(debrief)
          print("-----------------------------\n")

      except Exception as e:
        # Ignore temporary lookup errors while folders are being created
        pass

      time.sleep(self.poll_interval)


if __name__ == "__main__":
  engineer = RaceEngineer()
  engineer.start_listener()