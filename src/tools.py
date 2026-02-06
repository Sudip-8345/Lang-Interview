from langchain_core.tools.retriever import create_retriever_tool
from langchain_core.tools import tool
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, HRFlowable
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY
from reportlab.lib.colors import HexColor, black, white
from reportlab.lib.units import inch
from datetime import datetime
import os

from utils.logger import get_logger

logger = get_logger(__name__)


def create_jd_tool(jd_retriever):
    try:
        qa_tool = create_retriever_tool(
            retriever=jd_retriever,
            name="interview_document_retriever",
            description="Useful for answering questions about the interview document and finding relevant interview questions."
        )
        logger.info("JD retriever tool created successfully.")
        return qa_tool
    except Exception as e:
        logger.error(f"Error creating JD retriever tool: {e}")
        raise RuntimeError(f"Failed to create JD retriever tool: {e}")


def create_resume_tool(resume_retriever):
    try:
        resume_tool = create_retriever_tool(
            retriever=resume_retriever, 
            name="candidate_resume_retriever",
            description="Useful for answering questions about the candidate's resume, projects, and experience."
        )
        logger.info("Resume retriever tool created successfully.")
        return resume_tool
    except Exception as e:
        logger.error(f"Error creating resume retriever tool: {e}")
        raise RuntimeError(f"Failed to create resume retriever tool: {e}")


# --- PDF Report Tool ---
# Color scheme for reports
PRIMARY_COLOR = HexColor("#1a365d")
SECONDARY_COLOR = HexColor("#2b6cb0")
ACCENT_COLOR = HexColor("#38a169")
WARNING_COLOR = HexColor("#dd6b20")
BORDER_COLOR = HexColor("#e2e8f0")


@tool
def save_report_as_pdf(report_content: str, filename: str) -> str:
    """
    Saves the provided report content as a professionally styled PDF file.

    Args:
        report_content (str): The full text content of the HR report.
        filename (str): The desired name for the PDF file (e.g., "HR_Interview_Report.pdf").
                        Do NOT include path, just the filename.

    Returns:
        str: The full path to the saved PDF file if successful, otherwise an error message.
    """
    if not filename.endswith(".pdf"):
        filename += ".pdf"
    
    safe_filename = os.path.basename(filename)

    try:
        doc = SimpleDocTemplate(
            safe_filename, 
            pagesize=letter,
            rightMargin=0.75*inch,
            leftMargin=0.75*inch,
            topMargin=0.75*inch,
            bottomMargin=0.75*inch
        )
        styles = getSampleStyleSheet()
        
        # Custom styles
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Title'],
            fontSize=28,
            textColor=PRIMARY_COLOR,
            alignment=TA_CENTER,
            spaceAfter=6,
            fontName='Helvetica-Bold'
        )
        
        subtitle_style = ParagraphStyle(
            'CustomSubtitle',
            parent=styles['Normal'],
            fontSize=12,
            textColor=SECONDARY_COLOR,
            alignment=TA_CENTER,
            spaceAfter=20,
            fontName='Helvetica-Oblique'
        )
        
        section_header_style = ParagraphStyle(
            'SectionHeader',
            parent=styles['Heading2'],
            fontSize=14,
            textColor=white,
            backColor=PRIMARY_COLOR,
            borderPadding=(8, 8, 8, 8),
            spaceBefore=16,
            spaceAfter=10,
            fontName='Helvetica-Bold'
        )
        
        body_style = ParagraphStyle(
            'CustomBody',
            parent=styles['Normal'],
            fontSize=11,
            textColor=black,
            alignment=TA_JUSTIFY,
            spaceAfter=8,
            leading=16,
            fontName='Helvetica'
        )
        
        highlight_style = ParagraphStyle(
            'Highlight',
            parent=styles['Normal'],
            fontSize=11,
            textColor=ACCENT_COLOR,
            spaceAfter=6,
            fontName='Helvetica-Bold',
            leftIndent=15
        )
        
        warning_style = ParagraphStyle(
            'Warning',
            parent=styles['Normal'],
            fontSize=11,
            textColor=WARNING_COLOR,
            spaceAfter=6,
            fontName='Helvetica-Bold',
            leftIndent=15
        )

        story = []
        
        # Header
        story.append(Paragraph("📋 HR Interview Report", title_style))
        story.append(Paragraph(
            f"Generated on {datetime.now().strftime('%B %d, %Y at %I:%M %p')}", 
            subtitle_style
        ))
        
        story.append(HRFlowable(
            width="100%", thickness=2, color=PRIMARY_COLOR,
            spaceBefore=5, spaceAfter=20
        ))

        # Section keywords for styling
        section_keywords = {
            'overall suitability': '📊 Overall Suitability',
            'strengths': '✅ Strengths',
            'areas for development': '⚠️ Areas for Development',
            'weaknesses': '⚠️ Areas for Development',
            'technical skills': '💻 Key Technical Skills',
            'problem-solving': '🧩 Problem-Solving Approach',
            'communication': '💬 Communication Skills',
            'experience': '📁 Relevant Experience',
            'recommendations': '🎯 Recommendations',
            'candidate summary': '👤 Candidate Summary',
        }

        # Process content
        paragraphs = report_content.split('\n')
        
        for para_text in paragraphs:
            para_stripped = para_text.strip()
            if not para_stripped:
                continue
                
            is_section = False
            for keyword, emoji_title in section_keywords.items():
                if keyword in para_stripped.lower() and (
                    para_stripped.endswith(':') or len(para_stripped) < 60
                ):
                    story.append(Spacer(1, 10))
                    story.append(Paragraph(emoji_title, section_header_style))
                    is_section = True
                    break
            
            if not is_section:
                if para_stripped.startswith(('-', '•', '*', '–')):
                    clean_text = para_stripped.lstrip('-•*– ').strip()
                    if any(word in para_stripped.lower() for word in 
                           ['strength', 'excellent', 'strong', 'proficient', 'demonstrated']):
                        story.append(Paragraph(f"✓ {clean_text}", highlight_style))
                    elif any(word in para_stripped.lower() for word in 
                             ['improve', 'develop', 'weakness', 'gap', 'lacking']):
                        story.append(Paragraph(f"△ {clean_text}", warning_style))
                    else:
                        story.append(Paragraph(f"• {clean_text}", body_style))
                else:
                    story.append(Paragraph(para_stripped, body_style))

        # Footer
        story.append(Spacer(1, 30))
        story.append(HRFlowable(
            width="100%", thickness=1, color=BORDER_COLOR,
            spaceBefore=10, spaceAfter=10
        ))
        
        footer_style = ParagraphStyle(
            'Footer', parent=styles['Normal'], fontSize=9,
            textColor=HexColor("#718096"), alignment=TA_CENTER
        )
        story.append(Paragraph(
            "This report was automatically generated by AI Interview Assistant", 
            footer_style
        ))
        story.append(Paragraph("Confidential - For HR Use Only", footer_style))

        doc.build(story)
        logger.info(f"Report saved to: {os.path.abspath(safe_filename)}")
        return f"✅ Report successfully saved to: {os.path.abspath(safe_filename)}"
    except Exception as e:
        logger.error(f"Error saving report as PDF: {e}")
        return f"❌ Error saving report as PDF: {e}"


# Export tools for report writing
report_writer_tools = [save_report_as_pdf]
