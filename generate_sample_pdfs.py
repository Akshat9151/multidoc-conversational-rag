import os
from pathlib import Path
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak

OUTPUT_DIR = Path(__file__).resolve().parent / "data" / "sample_docs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

styles = getSampleStyleSheet()

title_style = ParagraphStyle(
    "DocTitle",
    parent=styles["Normal"],
    fontName="Helvetica-Bold",
    fontSize=20,
    leading=24,
    textColor=colors.HexColor("#1A365D"),
    spaceAfter=12,
)

h1_style = ParagraphStyle(
    "DocH1",
    parent=styles["Normal"],
    fontName="Helvetica-Bold",
    fontSize=14,
    leading=18,
    textColor=colors.HexColor("#2B6CB0"),
    spaceBefore=10,
    spaceAfter=6,
)

body_style = ParagraphStyle(
    "DocBody",
    parent=styles["Normal"],
    fontName="Helvetica",
    fontSize=10,
    leading=14,
    textColor=colors.HexColor("#2D3748"),
    spaceAfter=8,
)


def create_financial_report():
    pdf_path = OUTPUT_DIR / "acme_financial_annual_report_2025.pdf"
    doc = SimpleDocTemplate(str(pdf_path), pagesize=letter, rightMargin=54, leftMargin=54, topMargin=54, bottomMargin=54)
    story = []

    # --- Page 1 ---
    story.append(Paragraph("Acme Global Technologies — Annual Financial Report 2025", title_style))
    story.append(Paragraph("Published: January 15, 2026 | Document ID: FIN-2025-AG", body_style))
    story.append(Spacer(1, 10))
    story.append(Paragraph("1. Executive Summary & Financial Highlights", h1_style))
    story.append(Paragraph(
        "For the fiscal year ended December 31, 2025, Acme Global Technologies demonstrated unprecedented operational efficiency and revenue expansion. "
        "Consolidated full-year revenue reached $485.4 million, representing a 28.5% year-over-year increase compared to $377.7 million in fiscal year 2024. "
        "Gross margin expanded by 320 basis points to 68.4%, driven by strong margin contributions from high-margin Enterprise SaaS and automated AI workflow pipelines. "
        "Full-year Net Income totaled $72.8 million with an Operating Margin of 19.4% and an EBITDA margin of 24.5%.",
        body_style
    ))
    story.append(Paragraph(
        "Cash, cash equivalents, and short-term marketable securities stood at $142.6 million at year-end, up from $98.2 million at the close of 2024. "
        "The company generated $94.3 million in operating cash flow and returned $25.0 million to shareholders via disciplined share repurchases.",
        body_style
    ))
    story.append(PageBreak())

    # --- Page 2 ---
    story.append(Paragraph("2. Segment Revenue & Regional Performance", h1_style))
    story.append(Paragraph(
        "Acme Global operates across three primary business divisions: Enterprise SaaS, Cloud AI Infrastructure, and Professional Implementation Services. "
        "Enterprise SaaS remained our primary growth engine, contributing $290.2 million (59.8% of total revenue) with a Net Dollar Retention rate of 118%. "
        "Cloud AI Infrastructure revenue grew by 74.2% year-over-year to $125.0 million (25.8% of total revenue), reflecting intense enterprise adoption of our automated agent platform. "
        "Professional Services accounted for the remaining $70.2 million (14.4% of total revenue), primarily focused on enterprise integration and security hardening.",
        body_style
    ))

    table_data = [
        ["Division / Metric", "FY2024 ($M)", "FY2025 ($M)", "YoY Growth", "% of Total"],
        ["Enterprise SaaS", "$234.0M", "$290.2M", "+24.0%", "59.8%"],
        ["Cloud AI Infrastructure", "$71.7M", "$125.0M", "+74.2%", "25.8%"],
        ["Professional Services", "$72.0M", "$70.2M", "-2.5%", "14.4%"],
        ["Total Consolidated Revenue", "$377.7M", "$485.4M", "+28.5%", "100.0%"],
    ]
    t = Table(table_data, colWidths=[150, 85, 85, 80, 80])
    t.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#2B6CB0")),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 9),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 6),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E0")),
        ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 1), (-1, -1), 9),
    ]))
    story.append(t)
    story.append(Spacer(1, 10))
    story.append(Paragraph(
        "Regional breakdown: North America contributed $281.5 million (58%), Europe, Middle East and Africa (EMEA) generated $116.5 million (24%), "
        "and Asia-Pacific (APAC) contributed $87.4 million (18%).",
        body_style
    ))
    story.append(PageBreak())

    # --- Page 3 ---
    story.append(Paragraph("3. Research & Development Investments and Risk Analysis", h1_style))
    story.append(Paragraph(
        "Acme invested $65.4 million into Research and Development during FY2025, representing 13.5% of total revenue. "
        "Key R&D initiatives included low-latency multi-modal vector search indexing, proprietary document retrieval engines, and fine-tuning open-source LLM architectures. "
        "Capital expenditures (CapEx) were $22.1 million, largely directed toward specialized high-bandwidth GPU cluster leaseholds in Ashburn and Frankfurt.",
        body_style
    ))
    story.append(Paragraph(
        "Key Risk Factors: 1) Foreign currency fluctuations across EUR and GBP could adversely impact non-USD denominated contracts. "
        "2) Enterprise procurement cycles lengthened from an average of 68 days in FY2024 to 84 days in FY2025 due to rigorous enterprise AI safety and data governance audits. "
        "3) Third-party cloud vendor compute availability remains a monitored dependency.",
        body_style
    ))

    doc.build(story)
    print(f"Generated: {pdf_path}")


def create_technical_manual():
    pdf_path = OUTPUT_DIR / "cloudscale_technical_architecture_manual.pdf"
    doc = SimpleDocTemplate(str(pdf_path), pagesize=letter, rightMargin=54, leftMargin=54, topMargin=54, bottomMargin=54)
    story = []

    # --- Page 1 ---
    story.append(Paragraph("CloudScale Core Systems — Technical Architecture Manual", title_style))
    story.append(Paragraph("Version 4.2 | Engineering Systems Division | Confidential", body_style))
    story.append(Spacer(1, 10))
    story.append(Paragraph("1. Microservices Infrastructure & Service Mesh", h1_style))
    story.append(Paragraph(
        "The CloudScale platform operates on an enterprise-grade distributed microservices architecture hosted on Amazon EKS (Kubernetes 1.29). "
        "All inter-service communications are governed by Istio Service Mesh using mutual TLS (mTLS) with automated certificate rotation every 30 days. "
        "The platform guarantees a production availability Service Level Agreement (SLA) of 99.99% uptime, with an API gateway p95 latency target of under 25 milliseconds for authenticated requests.",
        body_style
    ))
    story.append(Paragraph(
        "Traffic ingress is managed by Envoy Proxy instances deployed across three availability zones. "
        "Global rate limiting enforces a maximum of 1,200 requests per minute per tenant API key, with token bucket bursting allowances up to 1,800 RPM. "
        "All asynchronous task dispatching relies on Apache Kafka clusters partitioned across 16 message brokers, ensuring zero-loss message ingestion at 50,000 events per second.",
        body_style
    ))
    story.append(PageBreak())

    # --- Page 2 ---
    story.append(Paragraph("2. Data Layer, Caching, and Vector Search Architecture", h1_style))
    story.append(Paragraph(
        "Primary transactional state is managed by Amazon Aurora PostgreSQL 15.4 clusters configured with multi-region active-passive replication. "
        "Read-heavy query workloads are offloaded to four read replicas per cluster, maintaining replication lag under 15 milliseconds. "
        "Distributed application caching is handled by a Redis Cluster (v7.2) with a 98.2% cache hit ratio and automated LRU key eviction policies.",
        body_style
    ))
    story.append(Paragraph(
        "The Vector Search Subsystem utilizes FAISS and pgvector indexes to perform nearest-neighbor semantic search over enterprise document embeddings. "
        "Embeddings are generated using the sentence-transformers/all-MiniLM-L6-v2 model producing 384-dimensional dense vectors. "
        "Vector indexes are serialized to persistent NVMe SSD volumes every 6 hours and hot-reloaded into RAM with a target search latency of under 8 milliseconds for top-10 nearest neighbor queries.",
        body_style
    ))
    story.append(PageBreak())

    # --- Page 3 ---
    story.append(Paragraph("3. Security, Compliance, and Disaster Recovery", h1_style))
    story.append(Paragraph(
        "CloudScale maintains full compliance with SOC 2 Type II, ISO/IEC 27001:2022, and HIPAA regulations. "
        "All persistent storage volumes are encrypted at rest using AES-256 via AWS KMS customer-managed keys (CMKs). "
        "Data in transit requires TLS 1.3 encryption across all internal and public network boundaries.",
        body_style
    ))
    story.append(Paragraph(
        "Disaster Recovery (DR) Objectives: Recovery Point Objective (RPO) is strictly under 5 minutes for database transactions, "
        "and Recovery Time Objective (RTO) is guaranteed under 30 minutes in the event of an entire primary region outage. "
        "Automated cross-region backups are taken every 4 hours and retained for 90 days in immutable WORM (Write Once, Read Many) S3 storage vaults.",
        body_style
    ))

    doc.build(story)
    print(f"Generated: {pdf_path}")


def create_hr_policy():
    pdf_path = OUTPUT_DIR / "apex_global_hr_and_operational_policy.pdf"
    doc = SimpleDocTemplate(str(pdf_path), pagesize=letter, rightMargin=54, leftMargin=54, topMargin=54, bottomMargin=54)
    story = []

    # --- Page 1 ---
    story.append(Paragraph("Apex Global — Employee Handbook & Operational Policies", title_style))
    story.append(Paragraph("Effective: Calendar Year 2026 | People Operations Department", body_style))
    story.append(Spacer(1, 10))
    story.append(Paragraph("1. Working Hours & Hybrid Work Model", h1_style))
    story.append(Paragraph(
        "Apex Global champions a flexible, outcome-driven work environment. The standard working schedule comprises 40 hours per week. "
        "Under our Hybrid Work Model, employees are required to work on-site at their designated regional office 2 days per week, "
        "with the remaining 3 days eligible for remote work from home. Standard core collaboration hours are between 10:00 AM and 4:00 PM local time, "
        "during which all team members must be reachable on Slack and available for synchronous collaborative meetings.",
        body_style
    ))
    story.append(Paragraph(
        "Full-time remote arrangements may be approved on a case-by-case basis by Department Vice Presidents for specialized engineering roles. "
        "Employees working remotely are entitled to a one-time home office setup reimbursement of up to $750 for ergonomic desks, chairs, and monitors.",
        body_style
    ))
    story.append(PageBreak())

    # --- Page 2 ---
    story.append(Paragraph("2. Paid Time Off (PTO), Medical, and Parental Leave", h1_style))
    story.append(Paragraph(
        "All full-time employees receive 24 days of Paid Annual Leave per calendar year, accrued monthly at the rate of 2.0 days per completed month of service. "
        "A maximum of 10 unused annual leave days may be carried forward into the subsequent calendar year; any excess beyond 10 days will lapse on December 31st. "
        "Additionally, employees are granted 12 days of Paid Sick and Medical Leave annually, which require a physician's certificate for absences exceeding 3 consecutive days.",
        body_style
    ))
    story.append(Paragraph(
        "Parental Leave Policy: Primary caregivers (including birth mothers and adoptive parents) are entitled to 26 weeks of fully paid maternity leave. "
        "Secondary caregivers (spouses and partners) are entitled to 4 weeks of fully paid paternity/partner leave. "
        "Employees also receive 10 paid public holidays per year according to their local country calendar.",
        body_style
    ))
    story.append(PageBreak())

    # --- Page 3 ---
    story.append(Paragraph("3. Professional Development, Travel & Equipment Allowances", h1_style))
    story.append(Paragraph(
        "Apex Global actively invests in continuous career growth. Every full-time employee is allotted an individual Professional Learning Budget of $1,500 per fiscal year. "
        "This stipend covers approved technical certifications (e.g., AWS, GCP, CKA, AI-ML credentials), university coursework, industry conference registrations, and technical book purchases. "
        "Reimbursements must be submitted through the expense portal within 30 days of payment along with itemized receipts and proof of completion.",
        body_style
    ))
    story.append(Paragraph(
        "Travel and Meals: When traveling on company business, domestic daily meal per-diem allowances are capped at $85 per day, and international per-diems are capped at $120 per day. "
        "Company-issued laptops (MacBook Pro 16-inch M3 or Dell XPS 15) remain company property and are eligible for hardware refresh cycles every 36 months.",
        body_style
    ))

    doc.build(story)
    print(f"Generated: {pdf_path}")


if __name__ == "__main__":
    create_financial_report()
    create_technical_manual()
    create_hr_policy()
