from teleassist.ingestion.parse import clean_pdf_pages, fix_extraction_artifacts


def test_pdf_cleanup_removes_repeated_digit_headers_and_standalone_page_numbers(caplog) -> None:
    caplog.set_level("INFO")
    header = "THE GAZETTE OF INDIA : EXTRAORDINARY [PART III—SEC.4]"
    footer = "Reliance Jio Infocomm Limited | Telecom Consumer Charter Page"
    body_texts = ["Body text about alpha.", "Body text about beta.", "Body text about gamma."]
    pages = [
        f"{header}\n{footer} {number} of 37\n{number}\n17 of 1933.\n{body}"
        for number, body in enumerate(body_texts, start=1)
    ]
    pages.append("Body text about delta.")

    cleaned = clean_pdf_pages(pages, document_name="jio_charter.pdf")

    assert all(header not in page for page in cleaned)
    assert all("Reliance Jio" not in page for page in cleaned)
    assert all(f"\n{number}\n" not in page for number, page in enumerate(cleaned, start=1))
    assert all("17 of 1933." in page for page in cleaned[:3])
    assert all(
        body in page
        for page, body in zip(cleaned, [*body_texts, "Body text about delta."], strict=True)
    )
    assert "PDF cleanup removed" in caplog.text
    assert "jio_charter.pdf" in caplog.text


def test_corpus_vocabulary_repairs_split_words_and_spaced_hyphens(caplog) -> None:
    caplog.set_level("INFO")
    units = [
        {"text": "S ervice starts with pre -paid setup a s per the form."},
        {
            "text": "The service is prepaid prepaid and rules apply as per as per. "
            "DoT DoT DoT. "
            "Are a valid phrase. "
            "this service form."
        },
        {"text": "Do T instructions are common in the corpus."},
    ]

    fixed, counts = fix_extraction_artifacts(units)

    assert fixed[0]["text"] == "Service starts with pre-paid setup as per the form."
    assert counts["S ervice -> Service"] == 1
    assert counts["pre -paid -> pre-paid"] == 1
    assert counts["a s -> as"] == 1
    assert "Are a valid phrase." in fixed[1]["text"]
    assert "DoT instructions" in fixed[2]["text"]
    assert "Top 20 corpus-vocabulary extraction fixes" in caplog.text
