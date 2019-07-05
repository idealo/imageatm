from imageatm.components import Evaluation


def run_evaluation(image_dir: str, job_dir: str, report_pdf: bool = False, report_html: bool = False, **kwargs):
    eval = Evaluation(image_dir=image_dir, job_dir=job_dir, **kwargs)
    eval.run(report_pdf=report_pdf, report_html=report_html)
