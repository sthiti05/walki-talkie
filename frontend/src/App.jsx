import { useState } from 'react';
import './App.css';
import FileUpload from './components/FileUpload';
import ChatInterface from './components/ChatInterface';

function App() {
  const [documentInfo, setDocumentInfo] = useState(null);

  const resetSession = () => {
    setDocumentInfo(null);
  };

  return (
    <div className="app-container">
      <header className="header animate-fade-in">
        <h1>Walki-Talkie AI</h1>
        <p>Interactive PDF Assistant powered by Gemini Insights</p>
      </header>

      <main className="main-content">
        {!documentInfo ? (
          <FileUpload onUploadComplete={setDocumentInfo} />
        ) : (
          <>
            {/* Optional: we could render FileUpload here differently if we want to change files, 
                for now let's just show chat, with a button to upload a new one. */}
            <div style={{ display: 'flex', flexDirection: 'column', width: '100%', gap: '1rem' }}>
              <div style={{ alignSelf: 'flex-start' }}>
                <button onClick={resetSession} className="glass-panel" style={{ padding: '0.5rem 1rem', fontSize: '0.85rem' }}>
                  ← Upload Different Document
                </button>
              </div>
              <ChatInterface documentInfo={documentInfo} />
            </div>
          </>
        )}
      </main>
    </div>
  );
}

export default App;
