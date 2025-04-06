# DataHacks
lowchart TD
    A[User enters a topic] --> B[Collect tweets over time window]
    B --> C[Cluster & analyze sentiments]
    B --> D[Track tweet volume & retweet speed]
    C --> E[Measure sentiment polarity/spread]
    D --> F[Compute growth rate of topic signals]
    E --> F
    F --> G[Feed into topic virality predictor]
    G --> H[Output: Likely Viral 🔥 or Not Likely Yet 💤]
