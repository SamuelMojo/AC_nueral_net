import ollama
import pandas as pd
from telemetry_parser import TelemetryParser


class PitRadioChatbox:

  def __init__(self, outputs_dir="outputs"):
    self.parser = TelemetryParser(outputs_dir=outputs_dir)
    # Pre-load the dataframe once when the chat session starts
    print("Loading latest telemetry into memory...")
    self.df, self.file_path = self.load_latest_dataframe()

  def load_latest_dataframe(self):
    try:
      file_path = self.parser.get_latest_parquet_file()
      return pd.read_parquet(file_path), file_path
    except FileNotFoundError:
      return None, None

  def query_telemetry_context(self):
    if self.df is None or self.df.empty:
      return "No telemetry data available yet."

    context = f"""
    - Total Frames in Run: {len(self.df)}
    - Peak Speed: {self.df['speed'].max():.2f} m/s (Mean: {self.df['speed'].mean():.2f} m/s)
    - Max Lateral G-Force: {self.df['accelY'].abs().max():.2f} G
    - Front-Left Tire Core Temp (Mean): {self.df['fl_tire_temperature_core'].mean():.1f}°C
    - Front-Right Tire Core Temp (Mean): {self.df['fr_tire_temperature_core'].mean():.1f}°C
    - Rear-Left Tire Core Temp (Mean): {self.df['rl_tire_temperature_core'].mean():.1f}°C
    - Rear-Right Tire Core Temp (Mean): {self.df['rr_tire_temperature_core'].mean():.1f}°C
    - Mean Brake Input: {self.df['brakeStatus'].mean():.2f}
    - Mean Throttle Input: {self.df['accStatus'].mean():.2f}
    - Mean Rear Slip Ratio: {self.df['tyre_slip_ratio_rr'].mean():.3f}
    - Off Track Occurred: {bool(self.df['out_of_track'].any())}
    """
    return context

  def ask_engineer(self, user_question):
    telemetry_context = self.query_telemetry_context()

    system_prompt = f"""
    You are an expert motorsport race engineer communicating over the pit radio during an Assetto Corsa AI training session. 
    Answer the driver or team member's question naturally, professionally, and in character. 
    Base your answers strictly on the following live telemetry extracted from the latest run ({self.file_path}):
    
    {telemetry_context}
    
    Keep responses concise and direct like a real pit wall radio transmission.
    """

    try:
      response = ollama.chat(
          model="llama3.1",
          messages=[
              {"role": "system", "content": system_prompt},
              {"role": "user", "content": user_question},
          ],
      )
      return response["message"]["content"]
    except Exception as e:
      return f"Radio transmission error: {e}"


if __name__ == "__main__":
  chatbox = PitRadioChatbox()
  print("=== PIT RADIO ACTIVE ===")
  print("Ask your race engineer anything about the car.")
  print("Type 'exit' or 'quit' to close the radio channel.\n")

  while True:
    try:
      user_input = input("Pit Wall > ")
      if user_input.lower() in ["exit", "quit"]:
        break
      if not user_input.strip():
        continue

      print("\nEngineer responding...")
      reply = chatbox.ask_engineer(user_input)
      print(f"\nRace Engineer: {reply}\n" + "-" * 50)

    except KeyboardInterrupt:
      break