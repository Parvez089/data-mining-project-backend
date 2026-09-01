from fastapi import FastAPI, UploadFile, File, Form
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

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
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
        # ১. ফাইল রিড করা (CSV অথবা Excel `.xlsx` / `.xls` হ্যান্ডেল করা)
        if file is not None:
            contents = await file.read()
            filename = file.filename.lower()
            
            if filename.endswith((".xlsx", ".xls")):
                df = pd.read_excel(io.BytesIO(contents))
            else:
                df = pd.read_csv(io.BytesIO(contents), encoding="ISO-8859-1")
        else:
            # যদি ফাইল সিলেক্ট না করা থাকে, ডামি ডেটাসেট তৈরি করা
            np.random.seed(42)
            n = 500
            df = pd.DataFrame({
                "CustomerID": range(15000, 15000 + n),
                "InvoiceDate": pd.date_range(start="2025-01-01", periods=n),
                "Quantity": np.random.randint(1, 20, n),
                "UnitPrice": np.random.uniform(5.0, 100.0, n)
            })

        # ২. ডেটাসেট ক্লিন ও RFM মেট্রিক্সে রূপান্তর
        if "CustomerID" in df.columns and "Quantity" in df.columns and "UnitPrice" in df.columns:
            df = df.dropna(subset=["CustomerID"])
            df["CustomerID"] = df["CustomerID"].astype(int)
            
            if "InvoiceDate" in df.columns:
                df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"], errors="coerce")
                reference_date = df["InvoiceDate"].max() + pd.Timedelta(days=1)
            else:
                reference_date = pd.Timestamp("2025-12-31")

            df["TotalPrice"] = df["Quantity"] * df["UnitPrice"]

            # RFM Aggregation
            rfm = df.groupby("CustomerID").agg({
                "InvoiceDate": lambda x: (reference_date - x.max()).days,
                "CustomerID": "count",
                "TotalPrice": "sum"
            }).rename(columns={
                "InvoiceDate": "Recency",
                "CustomerID": "Frequency",
                "TotalPrice": "Monetary"
            }).reset_index()
        else:
            rfm = pd.DataFrame({
                "CustomerID": range(17850, 18050),
                "Recency": np.random.randint(1, 100, 200),
                "Frequency": np.random.randint(1, 50, 200),
                "Monetary": np.random.uniform(50, 5000, 200)
            })

        # ৩. ফিচার স্কেলিং
        X = rfm[["Recency", "Frequency", "Monetary"]]
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # ৪. ডাইনামিক DBSCAN ট্রেনিং
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        clusters = dbscan.fit_predict(X_scaled)
        rfm["Cluster"] = clusters

        # ৫. মেট্রিক্স ক্যালকুলেশন
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

        # ৬. PCA (Dimensionality Reduction)
        pca = PCA(n_components=2)
        pca_coords = pca.fit_transform(X_scaled)
        
        pca_points = []
        for i in range(len(rfm)):
            pca_points.append({
                "x": float(pca_coords[i, 0]),
                "y": float(pca_coords[i, 1]),
                "cluster": int(clusters[i])
            })

        # ৭. টেবিল ডেটা ফরম্যাট
        table_data = []
        for i, row in rfm.head(150).iterrows():
            table_data.append({
                "CustomerID": int(row["CustomerID"]),
                "Recency": int(row["Recency"]),
                "Frequency": int(row["Frequency"]),
                "Monetary": float(row["Monetary"]),
                "Cluster": int(row["Cluster"])
            })

        print(f"Processed -> Customers: {total_customers}, Eps: {eps}, MinSamples: {min_samples}, Clusters: {clusters_found}, Outliers: {outliers}")

        return {
            "total_customers": total_customers,
            "clusters_found": clusters_found,
            "outliers": outliers,
            "silhouette_score": round(sil_score, 2),
            "pca_points": pca_points,
            "data": table_data
        }

    except Exception as e:
        print(f"Error in clustering: {str(e)}")
        return {
            "total_customers": 0,
            "clusters_found": 0,
            "outliers": 0,
            "silhouette_score": 0.0,
            "pca_points": [],
            "data": []
        }

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=True)