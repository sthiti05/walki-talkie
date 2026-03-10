import React, { useState, useRef } from 'react';
import './FileUpload.css';

export default function FileUpload({ onUploadComplete }) {
    const [isHovering, setIsHovering] = useState(false);
    const [isUploading, setIsUploading] = useState(false);
    const [error, setError] = useState(null);
    const fileInputRef = useRef(null);

    const handleDragOver = (e) => {
        e.preventDefault();
        setIsHovering(true);
    };

    const handleDragLeave = () => {
        setIsHovering(false);
    };

    const handleDrop = (e) => {
        e.preventDefault();
        setIsHovering(false);

        if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
            processFile(e.dataTransfer.files[0]);
        }
    };

    const handleFileChange = (e) => {
        if (e.target.files && e.target.files.length > 0) {
            processFile(e.target.files[0]);
        }
    };

    const processFile = async (file) => {
        setError(null);
        if (file.type !== 'application/pdf' && !file.name.toLowerCase().endsWith('.pdf')) {
            setError('Please upload a PDF file.');
            return;
        }

        setIsUploading(true);
        const formData = new FormData();
        formData.append('file', file);

        try {
            const response = await fetch('http://localhost:8000/api/upload', {
                method: 'POST',
                body: formData,
            });

            if (!response.ok) {
                const errorData = await response.json();
                throw new Error(errorData.detail || 'Upload failed');
            }

            const data = await response.json();
            onUploadComplete(data);
        } catch (err) {
            console.error('Upload Error:', err);
            setError(err.message || 'An error occurred during upload.');
        } finally {
            setIsUploading(false);
            if (fileInputRef.current) {
                fileInputRef.current.value = ''; // Reset input
            }
        }
    };

    return (
        <div className="upload-container glass-panel animate-fade-in">
            <h2>Upload PDF</h2>
            <p className="upload-subtitle">Select or drag a PDF document to begin analysis</p>

            <div
                className={`drop-zone ${isHovering ? 'hovering' : ''} ${error ? 'error' : ''}`}
                onDragOver={handleDragOver}
                onDragLeave={handleDragLeave}
                onDrop={handleDrop}
                onClick={() => fileInputRef.current?.click()}
            >
                <input
                    type="file"
                    ref={fileInputRef}
                    onChange={handleFileChange}
                    accept="application/pdf"
                    style={{ display: 'none' }}
                />

                {isUploading ? (
                    <div className="loading-state">
                        <div className="spinner"></div>
                        <p>Processing Document...</p>
                        <span className="loading-subtext">Extracting and chunking text</span>
                    </div>
                ) : (
                    <div className="ready-state">
                        <svg xmlns="http://www.w3.org/2000/svg" width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className="upload-icon">
                            <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"></path>
                            <polyline points="17 8 12 3 7 8"></polyline>
                            <line x1="12" y1="3" x2="12" y2="15"></line>
                        </svg>
                        <h3>Click or Drag to Upload</h3>
                        <p>PDF files only (max 50MB)</p>
                    </div>
                )}
            </div>

            {error && <div className="error-message">{error}</div>}
        </div>
    );
}