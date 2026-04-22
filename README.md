# Live Math Grade Predictor

A Streamlit classroom demo with:
- presenter dashboard
- QR-code audience voting page
- live shared votes
- freeze-and-predict workflow
- interactive charts and tables

Run:
```bash
pip install -r requirements.txt
streamlit run app.py
```

Audience page:
- same app with `?mode=vote`

Note:
- shared state is stored in a local SQLite file
- this is ideal for a classroom demo
