import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score

def process_customer_segmentation(file_file, eps: float, min_samples: int):
    df = pd.read_csv(file_file, encoding='ISO-8859-1')
    
    # ইউনিভার্সাল কলাম ম্যাপিং (ভিন্ন নামের কলাম হ্যান্ডেল করার জন্য)
    column_aliases = {
        'CustomerID': ['CustomerID', 'Customer_ID', 'Client_ID', 'User_ID', 'ID'],
        'Quantity': ['Quantity', 'Qty', 'Amount', 'Units'],
        'UnitPrice': ['UnitPrice', 'Price', 'Rate', 'Cost'],
        'InvoiceDate': ['InvoiceDate', 'Date', 'TransactionDate', 'Timestamp'],
        'InvoiceNo': ['InvoiceNo', 'Invoice_No', 'BillNo', 'TransactionID']
    }

    for standard_name, aliases in column_aliases.items():
        matched = False
        for alias in aliases:
            if alias in df.columns:
                df.rename(columns={alias: standard_name}, inplace=True)
                matched = True
                break
        if not matched and standard_name != 'InvoiceNo':
            # যদি কলামটি একদম না থাকে তবে ডিফল্ট বা ডামি ভ্যালু তৈরি করা যেতে পারে
            pass

    # TotalPrice ক্যালকুলেশন
    if 'TotalPrice' not in df.columns:
        if 'Quantity' in df.columns and 'UnitPrice' in df.columns:
            df['TotalPrice'] = df['Quantity'] * df['UnitPrice']
        else:
            df['TotalPrice'] = 1.0  # ফলব্যাক ভ্যালু

    if 'CustomerID' in df.columns:
        df = df.dropna(subset=['CustomerID'])
        
    if 'InvoiceDate' in df.columns:
        df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'], errors='coerce')
        snapshot_date = df['InvoiceDate'].max() + pd.Timedelta(days=1)
    else:
        snapshot_date = pd.Timestamp.now()
        df['InvoiceDate'] = snapshot_date

    # RFM ক্যালকুলেশন
    rfm = df.groupby('CustomerID').agg({
        'InvoiceDate': lambda x: (snapshot_date - x.max()).days if pd.notnull(x.max()) else 1,
        'InvoiceNo': 'count' if 'InvoiceNo' in df.columns else lambda x: len(x),
        'TotalPrice': 'sum'
    }).reset_index()
    rfm.columns = ['CustomerID', 'Recency', 'Frequency', 'Monetary']

    # ফিচার স্কেলিং ও PCA
    X = rfm[['Recency', 'Frequency', 'Monetary']]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)

    # DBSCAN মডেল রান
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    clusters = dbscan.fit_predict(X_scaled)
    rfm['Cluster'] = clusters

    # সিল্যুয়েট স্কোর হিসাব
    labels = dbscan.labels_
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    sil_score = 0.0
    if n_clusters > 1 and len(set(labels)) > 1:
        try:
            sil_score = float(silhouette_score(X_scaled, labels))
        except:
            sil_score = 0.0

    return {
        "total_customers": len(rfm),
        "clusters_found": n_clusters,
        "outliers": list(clusters).count(-1),
        "silhouette_score": round(sil_score, 3),
        "data": rfm.head(50).to_dict(orient="records"),
        "pca_coords": [{"x": float(p[0]), "y": float(p[1]), "cluster": int(c)} for p, c in zip(X_pca, clusters)]
    }