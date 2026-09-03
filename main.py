from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import io
import os
import uvicorn

app = FastAPI(title="SegmentPulse AI Backend", version="2.6")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "https://data-mining-project-frontend.vercel.app"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/api/cluster")
async def run_cluster(
    file: UploadFile = File(None),
    eps: float = Form(1.2),
    min_samples: int = Form(8)
):
    try:
        if file is not None and file.filename:
            contents = await file.read()
            filename = file.filename.lower()
            
            if filename.endswith((".xlsx", ".xls")):
                df = pd.read_excel(io.BytesIO(contents))
            else:
                df = pd.read_csv(io.BytesIO(contents), encoding="ISO-8859-1")
            
            df.columns = df.columns.str.strip()
        else:
            np.random.seed(42)
            n = 500
            df = pd.DataFrame({
                "CustomerID": range(15000, 15000 + n),
                "InvoiceDate": pd.date_range(start="2025-01-01", periods=n),
                "Quantity": np.random.randint(1, 20, n),
                "UnitPrice": np.random.uniform(5.0, 100.0, n)
            })

        # Flexible column mapping handling variations, spaces, and truncated CSV/Excel headers
        cols_lower = {c.lower().replace("_", ""): c for c in df.columns}
        
        cust_col = (
            cols_lower.get("customerid") or 
            cols_lower.get("customer id") or 
            cols_lower.get("customer")
        )
        qty_col = cols_lower.get("quantity")
        price_col = (
            cols_lower.get("unitprice") or 
            cols_lower.get("unit price") or 
            cols_lower.get("price")
        )
        date_col = (
            cols_lower.get("invoicedate") or 
            cols_lower.get("invoice date") or 
            cols_lower.get("invoiceda") or 
            cols_lower.get("date")
        )
        invoice_no_col = (
            cols_lower.get("invoiceno") or 
            cols_lower.get("invoice no") or 
            cols_lower.get("invoice")
        )

        if cust_col and qty_col and price_col:
            df = df.dropna(subset=[cust_col])
            df[cust_col] = pd.to_numeric(df[cust_col], errors="coerce")
            df = df.dropna(subset=[cust_col])
            df[cust_col] = df[cust_col].astype(int)
            
            if date_col:
                df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
                valid_dates = df[date_col].dropna()
                reference_date = valid_dates.max() + pd.Timedelta(days=1) if not valid_dates.empty else pd.Timestamp("2025-12-31")
            else:
                reference_date = pd.Timestamp("2025-12-31")

            qty_series = pd.to_numeric(df[qty_col], errors="coerce").fillna(0)
            price_series = pd.to_numeric(df[price_col], errors="coerce").fillna(0)
            df["TotalPrice"] = qty_series * price_series

            # RFM Grouping execution
            freq_col_name = invoice_no_col if invoice_no_col else qty_series.name
            agg_rules = {
                freq_col_name: "nunique" if invoice_no_col else "count",
                "TotalPrice": "sum"
            }
            if date_col:
                agg_rules[date_col] = lambda x: (reference_date - x.max()).days if not x.isna().all() else 30

            rfm = df.groupby(cust_col).agg(agg_rules).reset_index()
            
            # Rename columns safely based on available keys
            rename_map = {cust_col: "CustomerID", "TotalPrice": "Monetary"}
            if date_col:
                rename_map[date_col] = "Recency"
                rename_map[freq_col_name] = "Frequency"
            else:
                rfm["Recency"] = 30
                rename_map[freq_col_name] = "Frequency"

            rfm = rfm.rename(columns=rename_map)
        else:
            rfm = pd.DataFrame({
                "CustomerID": range(17850, 18050),
                "Recency": np.random.randint(1, 100, 200),
                "Frequency": np.random.randint(1, 50, 200),
                "Monetary": np.random.uniform(50, 5000, 200)
            })

        rfm = rfm.replace([np.inf, -np.inf], np.nan).dropna(subset=["Recency", "Frequency", "Monetary"])

        if len(rfm) < min_samples:
            raise HTTPException(status_code=400, detail=f"Dataset contains insufficient valid customer records ({len(rfm)}) for min_samples={min_samples}.")

        X = rfm[["Recency", "Frequency", "Monetary"]]
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        clusters = dbscan.fit_predict(X_scaled)
        rfm["Cluster"] = clusters

        total_customers = int(len(rfm))
        clusters_found = int(len(set(clusters)) - (1 if -1 in clusters else 0))
        outliers = int(list(clusters).count(-1))

        try:
            valid_mask = clusters != -1
            if len(set(clusters[valid_mask])) > 1:
                sil_score = float(silhouette_score(X_scaled[valid_mask], clusters[valid_mask]))
            else:
                sil_score = 0.0
        except Exception:
            sil_score = 0.0

        pca = PCA(n_components=2)
        pca_coords = pca.fit_transform(X_scaled)
        
        pca_points = [
            {
                "x": float(pca_coords[i, 0]),
                "y": float(pca_coords[i, 1]),
                "cluster": int(clusters[i])
            }
            for i in range(len(rfm))
        ]

        table_data = [
            {
                "CustomerID": int(row["CustomerID"]),
                "Recency": int(row["Recency"]),
                "Frequency": int(row["Frequency"]),
                "Monetary": float(row["Monetary"]),
                "Cluster": int(row["Cluster"])
            }
            for i, row in rfm.head(150).iterrows()
        ]

        return {
            "total_customers": total_customers,
            "clusters_found": clusters_found,
            "outliers": outliers,
            "silhouette_score": round(sil_score, 2),
            "pca_points": pca_points,
            "data": table_data
        }

    except HTTPException as he:
        raise he
    except Exception as e:
        print(f"Error in clustering pipeline: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal clustering processing error: {str(e)}")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=True)