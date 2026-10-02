"""
features.py - SANNIF Feature Extensions Module
Provides: attack info database, PDF generator, demo dataset loader,
          health score, risk gauge data, severity badges, recommendations.
"""

import os
import io
import datetime
import pandas as pd
import numpy as np


# ---------------------------------------------------------------------------
# Attack Information Database
# ---------------------------------------------------------------------------
ATTACK_INFO = {
    "Normal": {
        "icon": "GREEN_CIRCLE",
        "severity": "None",
        "severity_badge": "low",
        "description": "Standard secure network transaction with no anomalous characteristics detected.",
        "impact": "No impact. This traffic represents legitimate, expected network behaviour.",
        "mitigation": "No action required. Continue monitoring as part of normal baseline traffic.",
        "prevention": [
            "Maintain up-to-date network traffic baselines.",
            "Periodically re-train the model on fresh normal traffic samples.",
            "Ensure logging is enabled even for normal sessions for forensic readiness.",
        ]
    },
    "Generic": {
        "icon": "ORANGE_CIRCLE",
        "severity": "High",
        "severity_badge": "high",
        "description": "Non-specific anomalous packet structures that deviate from normal baselines without matching a known signature.",
        "impact": "Potential reconnaissance or unknown zero-day activity. May serve as a precursor to a targeted attack.",
        "mitigation": "Isolate affected network segments. Capture packets for forensic analysis. Alert Tier-2 SOC analysts.",
        "prevention": [
            "Deploy anomaly-based IDS alongside signature-based systems.",
            "Implement strict egress filtering on all outbound ports.",
            "Enable deep packet inspection (DPI) on edge routers.",
            "Rate-limit connections from unfamiliar IP ranges.",
        ]
    },
    "Exploits": {
        "icon": "RED_CIRCLE",
        "severity": "Critical",
        "severity_badge": "critical",
        "description": "Attacks leveraging known software vulnerabilities to execute unauthorized code or gain elevated privileges.",
        "impact": "Full system compromise, remote code execution, privilege escalation, data exfiltration.",
        "mitigation": "Immediately isolate affected systems. Apply emergency patches. Revoke compromised credentials. Perform memory forensics.",
        "prevention": [
            "Maintain strict patch management (< 72h for critical CVEs).",
            "Deploy Web Application Firewalls (WAF) and host-based intrusion prevention (HIPS).",
            "Enforce principle of least privilege across all services.",
            "Conduct regular vulnerability assessments and penetration testing.",
        ]
    },
    "Fuzzers": {
        "icon": "ORANGE_CIRCLE",
        "severity": "High",
        "severity_badge": "high",
        "description": "Automated probing sending malformed or boundary-value inputs to crash services or find vulnerabilities.",
        "impact": "Service instability, denial of availability, discovery of 0-day vulnerabilities.",
        "mitigation": "Block source IPs generating high-volume malformed requests. Enable application-level rate limiting.",
        "prevention": [
            "Implement robust input validation on all public-facing endpoints.",
            "Use API gateways with request rate limiting and anomaly detection.",
            "Deploy honeypots to detect and study fuzzing campaigns.",
            "Harden network services against malformed protocol messages.",
        ]
    },
    "DoS": {
        "icon": "RED_CIRCLE",
        "severity": "Critical",
        "severity_badge": "critical",
        "description": "Denial of Service attacks designed to exhaust server resources making services unavailable.",
        "impact": "Complete service unavailability, financial loss, SLA violations, reputational damage.",
        "mitigation": "Activate DDoS mitigation provider. Rate limit at the load balancer. Enable auto-scaling.",
        "prevention": [
            "Use a cloud-based DDoS protection service with always-on scrubbing.",
            "Implement SYN cookies on TCP stacks to prevent SYN flood attacks.",
            "Configure connection limits and timeouts on all public services.",
            "Establish an incident response playbook for DoS scenarios.",
        ]
    },
    "Reconnaissance": {
        "icon": "YELLOW_CIRCLE",
        "severity": "Medium",
        "severity_badge": "medium",
        "description": "Systematic port sweeps, OS fingerprinting, and service enumeration to map a target network before an attack.",
        "impact": "Intelligence gathering enabling more sophisticated follow-on attacks.",
        "mitigation": "Block scanning source IPs. Review firewall rules. Ensure sensitive ports are not externally exposed.",
        "prevention": [
            "Deploy a network scanner detection system (e.g. Portsentry).",
            "Implement firewalls with port-knocking or stealth configurations.",
            "Limit ICMP responses and banner information disclosure.",
            "Use network segmentation to limit the blast radius of reconnaissance.",
        ]
    },
    "Analysis": {
        "icon": "YELLOW_CIRCLE",
        "severity": "Medium",
        "severity_badge": "medium",
        "description": "Traffic analysis attacks involving port scanning, HTML file transfers with embedded payloads, or spam infiltration.",
        "impact": "Data leakage, phishing payload delivery, network mapping, user credential exposure.",
        "mitigation": "Filter suspicious file transfers at the email/web gateway. Review inbound HTML content.",
        "prevention": [
            "Enable email content scanning and sandboxing for attachments.",
            "Deploy SSL/TLS inspection on egress traffic.",
            "Use DNS sinkholes to prevent communication with known malicious domains.",
            "Train staff to identify social engineering and phishing attempts.",
        ]
    },
    "Backdoor": {
        "icon": "RED_CIRCLE",
        "severity": "Critical",
        "severity_badge": "critical",
        "description": "Rogue access mechanisms installed to maintain persistent, hidden access to compromised systems.",
        "impact": "Persistent attacker presence, ongoing data theft, system integrity compromise, lateral movement.",
        "mitigation": "Immediately isolate the affected host. Perform full disk forensics. Rebuild from a known-good image.",
        "prevention": [
            "Implement file integrity monitoring (FIM) on all critical system files.",
            "Use application whitelisting to prevent unauthorized executable installation.",
            "Conduct regular outbound traffic analysis to detect C2 communication.",
            "Enforce multi-factor authentication on all administrative access.",
        ]
    },
    "Shellcode": {
        "icon": "RED_CIRCLE",
        "severity": "Critical",
        "severity_badge": "critical",
        "description": "Small payloads of machine code injected into vulnerable processes to spawn shell sessions or execute arbitrary commands.",
        "impact": "Immediate code execution, privilege escalation, system takeover, lateral movement.",
        "mitigation": "Enable DEP and ASLR. Patch vulnerable services immediately. Perform memory forensics.",
        "prevention": [
            "Enable hardware-enforced DEP on all servers.",
            "Compile applications with stack canaries and ASLR support.",
            "Deploy runtime application self-protection (RASP) on critical services.",
            "Monitor for unusual process spawning patterns using EDR solutions.",
        ]
    },
    "Worms": {
        "icon": "RED_CIRCLE",
        "severity": "Critical",
        "severity_badge": "critical",
        "description": "Self-replicating malicious software that propagates across network hosts autonomously.",
        "impact": "Rapid network-wide compromise, bandwidth exhaustion, mass data encryption, supply chain contamination.",
        "mitigation": "Immediately segment the network. Block lateral movement. Deploy emergency AV/EDR signatures.",
        "prevention": [
            "Enforce network micro-segmentation to limit lateral propagation.",
            "Disable unnecessary file-sharing protocols (SMB, RDP) or harden with MFA.",
            "Apply critical OS patches immediately.",
            "Implement behavioural endpoint detection to catch self-replication patterns.",
        ]
    },
}

SEVERITY_BADGE_STYLES = {
    "low":      {"bg": "#DCFCE7", "color": "#15803D", "border": "#BBF7D0", "label": "Low"},
    "medium":   {"bg": "#FEF3C7", "color": "#B45309", "border": "#FDE68A", "label": "Medium"},
    "high":     {"bg": "#FFEDD5", "color": "#C2410C", "border": "#FED7AA", "label": "High"},
    "critical": {"bg": "#FEE2E2", "color": "#B91C1C", "border": "#FECACA", "label": "Critical"},
}

SEVERITY_EMOJI = {
    "low": "🟢", "medium": "🟡", "high": "🟠", "critical": "🔴"
}


def get_severity_badge_html(attack_name: str) -> str:
    info = ATTACK_INFO.get(attack_name, ATTACK_INFO["Generic"])
    key = info["severity_badge"]
    style = SEVERITY_BADGE_STYLES[key]
    emoji = SEVERITY_EMOJI[key]
    return (
        f'<span style="display:inline-block;padding:2px 10px;border-radius:6px;'
        f'font-size:0.75rem;font-weight:700;background:{style["bg"]};'
        f'color:{style["color"]};border:1px solid {style["border"]};">'
        f'{emoji} {style["label"]}</span>'
    )


def compute_health_score(kpi: dict) -> int:
    total = kpi.get("total_records", 0)
    if total == 0:
        return 100
    threat_pct = (kpi.get("detected_attacks", 0) / total) * 100
    avg_threat_conf = kpi.get("avg_threat_confidence", 0.75) * 100
    score = 100 - (threat_pct * 1.2) - (avg_threat_conf * 0.1 if threat_pct > 0 else 0)
    return max(0, min(100, int(score)))


def compute_risk_level(kpi: dict) -> tuple:
    total = kpi.get("total_records", 0)
    if total == 0:
        return ("Unknown", "#86868B", "⚪")
    threat_pct = (kpi.get("detected_attacks", 0) / total) * 100
    if threat_pct == 0:
        return ("Low", "#34C759", "🟢")
    elif threat_pct < 10:
        return ("Low", "#34C759", "🟢")
    elif threat_pct < 30:
        return ("Medium", "#FF9500", "🟡")
    else:
        return ("High", "#FF3B30", "🔴")


def generate_recommendations(kpi: dict, pred_df: pd.DataFrame) -> list:
    recs = []
    total = kpi.get("total_records", 0)
    if total == 0:
        return ["Run detection on a dataset to generate recommendations."]
    attacks = pred_df[pred_df["Predicted Attack"] != "Normal"]["Predicted Attack"].value_counts()
    threat_pct = (kpi.get("detected_attacks", 0) / total) * 100

    if threat_pct == 0:
        recs.append("**No threats detected.** Continue monitoring and re-run analysis on fresh traffic samples regularly.")
        recs.append("**Baseline Documentation:** Document this clean traffic snapshot as a security baseline for future comparison.")
        return recs

    recs.append(f"**{threat_pct:.1f}% of traffic flagged as malicious.** Initiate your Incident Response Plan immediately if this is live traffic.")

    critical_types = {"DoS", "Exploits", "Worms", "Backdoor", "Shellcode"}
    if any(t in attacks.index for t in critical_types):
        recs.append("**Critical-severity threats detected.** Isolate affected network segments and escalate to Tier-3 SOC analysts within 15 minutes.")

    if "Reconnaissance" in attacks.index or "Analysis" in attacks.index:
        recs.append("**Reconnaissance activity detected.** An attacker may be mapping your network. Review exposed ports and disable unnecessary services.")

    if "Generic" in attacks.index or "Fuzzers" in attacks.index:
        recs.append("**Fuzzing/Generic anomalies detected.** Deploy WAF rules and enable rate limiting on all public-facing APIs.")

    if "Backdoor" in attacks.index or "Shellcode" in attacks.index:
        recs.append("**Backdoor/Shellcode detected.** Assume full host compromise. Perform forensic disk imaging before remediation and rotate all credentials.")

    if "Worms" in attacks.index:
        recs.append("**Worm activity detected.** Enable network micro-segmentation NOW to prevent lateral propagation. Patch all unpatched OS vulnerabilities.")

    recs.append("**Firewall Action:** Block all traffic from source IPs associated with flagged packets. Enable geo-blocking for regions with no business justification.")
    recs.append("**Monitoring Enhancement:** Increase log retention to 90 days and forward all SANNIF flagged events to your SIEM platform for correlation.")
    return recs[:7]


def load_demo_dataset(n_rows: int = 200) -> pd.DataFrame:
    dataset_path = os.path.join("models", "encoded_dataset.csv")
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Demo dataset not found at '{dataset_path}'.")
    df = pd.read_csv(dataset_path, nrows=n_rows)
    return df


def generate_pdf_report(kpi: dict, pred_df: pd.DataFrame) -> bytes:
    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.lib import colors
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.units import mm
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable
        from reportlab.lib.enums import TA_CENTER, TA_LEFT
    except ImportError:
        raise ImportError("reportlab is required. Run: pip install reportlab")

    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4, rightMargin=20*mm, leftMargin=20*mm, topMargin=20*mm, bottomMargin=20*mm)
    styles = getSampleStyleSheet()
    story = []

    BLUE   = colors.HexColor("#007AFF")
    GREEN  = colors.HexColor("#34C759")
    RED    = colors.HexColor("#FF3B30")
    ORANGE = colors.HexColor("#FF9500")
    DARK   = colors.HexColor("#1D1D1F")
    LIGHT  = colors.HexColor("#F5F5F7")
    MUTED  = colors.HexColor("#86868B")

    h1   = ParagraphStyle("h1", fontName="Helvetica-Bold", fontSize=20, textColor=BLUE, spaceAfter=4, alignment=TA_CENTER)
    h2   = ParagraphStyle("h2", fontName="Helvetica-Bold", fontSize=13, textColor=DARK, spaceAfter=6, spaceBefore=14)
    body = ParagraphStyle("body", fontName="Helvetica", fontSize=10, textColor=DARK, spaceAfter=4, leading=14)
    muted_s = ParagraphStyle("muted", fontName="Helvetica", fontSize=9, textColor=MUTED, spaceAfter=2, alignment=TA_CENTER)

    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    risk_label, _, risk_emoji = compute_risk_level(kpi)
    health = compute_health_score(kpi)
    total  = kpi.get("total_records", 0)
    threats = kpi.get("detected_attacks", 0)
    normal  = kpi.get("normal_traffic", 0)
    threat_pct = (threats / total * 100) if total > 0 else 0

    story.append(Paragraph("SANNIF Security Intelligence Report", h1))
    story.append(Paragraph("Semantic AI Network Intrusion Framework  |  Powered by XGBoost Baseline  |  UNSW-NB15", muted_s))
    story.append(Paragraph(f"Generated: {ts}", muted_s))
    story.append(HRFlowable(width="100%", thickness=1, color=BLUE, spaceAfter=10))

    risk_color = GREEN if risk_label == "Low" else (ORANGE if risk_label == "Medium" else RED)
    banner_data = [[
        Paragraph("Overall Risk", ParagraphStyle("bk", fontName="Helvetica", fontSize=9, textColor=colors.white)),
        Paragraph(f"{risk_emoji} {risk_label}", ParagraphStyle("bv", fontName="Helvetica-Bold", fontSize=16, textColor=colors.white)),
        Paragraph("Health Score", ParagraphStyle("bk", fontName="Helvetica", fontSize=9, textColor=colors.white)),
        Paragraph(f"{health}/100", ParagraphStyle("bv", fontName="Helvetica-Bold", fontSize=16, textColor=colors.white)),
        Paragraph("Threat Rate", ParagraphStyle("bk", fontName="Helvetica", fontSize=9, textColor=colors.white)),
        Paragraph(f"{threat_pct:.1f}%", ParagraphStyle("bv", fontName="Helvetica-Bold", fontSize=16, textColor=colors.white)),
    ]]
    bt = Table(banner_data, colWidths=[30*mm, 35*mm, 30*mm, 30*mm, 28*mm, 27*mm])
    bt.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), risk_color),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 8),
        ("TOPPADDING", (0, 0), (-1, -1), 10),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 10),
    ]))
    story.append(bt)
    story.append(Spacer(1, 10))

    story.append(Paragraph("Executive KPI Summary", h2))
    det_time = kpi.get("detection_time", 0)
    recs_per_sec = f"{(total / det_time):.0f} rec/s" if det_time > 0 else "N/A"
    kpi_rows = [
        ["Metric", "Value"],
        ["Total Records Analyzed", f"{total:,}"],
        ["Normal Traffic", f"{normal:,} ({(normal/total*100) if total>0 else 0:.1f}%)"],
        ["Threats Detected", f"{threats:,} ({threat_pct:.1f}%)"],
        ["Highest Risk Attack", str(kpi.get("highest_risk_attack", "None"))],
        ["Average Confidence", f"{kpi.get('avg_confidence', 0):.2f}%"],
        ["Detection Time", f"{det_time*1000:.1f} ms"],
        ["Records per Second", recs_per_sec],
        ["Model", "XGBoost Baseline"],
        ["Dataset", "UNSW-NB15"],
    ]
    kt = Table(kpi_rows, colWidths=[90*mm, 80*mm])
    kt.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), BLUE),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 10),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, LIGHT]),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#E5E5EA")),
        ("LEFTPADDING", (0, 0), (-1, -1), 8),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
    ]))
    story.append(kt)
    story.append(Spacer(1, 10))

    story.append(Paragraph("Threat Classification Distribution", h2))
    dist = pred_df["Predicted Attack"].value_counts().reset_index()
    dist.columns = ["Attack Category", "Count"]
    dist["Pct"] = (dist["Count"] / total * 100).round(2)
    dist_rows = [["Attack Category", "Count", "Percentage", "Severity"]]
    for _, row in dist.iterrows():
        info = ATTACK_INFO.get(row["Attack Category"], ATTACK_INFO["Generic"])
        dist_rows.append([row["Attack Category"], str(row["Count"]), f"{row['Pct']:.2f}%", info["severity"]])
    dt = Table(dist_rows, colWidths=[60*mm, 30*mm, 35*mm, 45*mm])
    dt.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), DARK),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, LIGHT]),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#E5E5EA")),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ]))
    story.append(dt)
    story.append(Spacer(1, 10))

    story.append(Paragraph("Security Recommendations", h2))
    recs = generate_recommendations(kpi, pred_df)
    for rec in recs:
        clean = rec.replace("**", "")
        story.append(Paragraph(f"- {clean}", body))

    story.append(Spacer(1, 16))
    story.append(HRFlowable(width="100%", thickness=0.5, color=MUTED))
    story.append(Paragraph(f"SANNIF v1.0  |  Confidential Security Intelligence Report  |  {ts}",
        ParagraphStyle("footer", fontName="Helvetica", fontSize=8, textColor=MUTED, alignment=TA_CENTER, spaceBefore=6)))

    doc.build(story)
    buffer.seek(0)
    return buffer.read()
