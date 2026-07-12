from app.pipeline import IndexingPipeline

pipeline = IndexingPipeline()
pipeline.index_pdf("data/raw/employee_handbook.pdf")