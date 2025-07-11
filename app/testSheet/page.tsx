// // pages/testSheet.tsx or app/testSheet/page.tsx (depending on your Next.js version)


// import { useEffect, useState } from 'react';
// import { useRouter } from 'next/navigation';
// import Head from 'next/head';

// interface SheetMusicPageProps {}

// const SheetMusicPage: React.FC<SheetMusicPageProps> = () => {
//     const [verovioToolkit, setVerovioToolkit] = useState<any>(null);
//     const [currentSvg, setCurrentSvg] = useState<string>('');
//     const [currentFileName, setCurrentFileName] = useState<string>('');
//     const [isInitialized, setIsInitialized] = useState<boolean>(false);
//     const [status, setStatus] = useState<{ message: string; type: string }>({ message: '', type: '' });
//     const [musicXMLData, setMusicXMLData] = useState<string | null>(null);
    
//     const router = useRouter();
    
//     // Load MusicXML data from localStorage or URL params
//     useEffect(() => {
//         const storedMusicXML = localStorage.getItem('currentMusicXML');
//         const storedFilename = localStorage.getItem('currentMusicXMLFilename');
        
//         if (storedMusicXML) {
//             setMusicXMLData(storedMusicXML);
//             setCurrentFileName(storedFilename || 'transcription');
//         }
        
//         // Alternative: Get from URL parameters
//         const { musicxml } = router.query;
//         if (musicxml && typeof musicxml === 'string') {
//             try {
//                 const decodedXML = decodeURIComponent(musicxml);
//                 setMusicXMLData(decodedXML);
//             } catch (error) {
//                 console.error('Error decoding MusicXML from URL:', error);
//             }
//         }
//     }, [router.query]);
    
//     // Initialize Verovio
//     useEffect(() => {
//         const initVerovio = async () => {
//             try {
//                 showStatus('Loading Verovio toolkit...', 'info');
                
//                 // Wait for Verovio to be available
//                 if (typeof window !== 'undefined' && (window as any).verovio) {
//                     const verovioModule = await (window as any).verovio.module;
//                     const toolkit = new (window as any).verovio.toolkit();
                    
//                     setVerovioToolkit(toolkit);
//                     setIsInitialized(true);
                    
//                     const version = toolkit.getVersion();
//                     showStatus(`Verovio toolkit v${version} loaded successfully!`, 'success');
                    
//                     // Auto-convert if MusicXML data is available
//                     if (musicXMLData) {
//                         convertMusicXML(musicXMLData, toolkit);
//                     }
//                 } else {
//                     throw new Error('Verovio library not found');
//                 }
//             } catch (error) {
//                 console.error('Failed to initialize Verovio:', error);
//                 showStatus(`Failed to initialize Verovio: ${typeof error === 'object' && error !== null && 'message' in error ? (error as { message: string }).message : String(error)}`, 'error');
//             }
//         };
        
//         // Add a delay to ensure Verovio script is loaded
//         const timer = setTimeout(initVerovio, 1000);
//         return () => clearTimeout(timer);
//     }, [musicXMLData]);
    
//     const showStatus = (message: string, type: string = 'info') => {
//         setStatus({ message, type });
//         if (type === 'success') {
//             setTimeout(() => setStatus({ message: '', type: '' }), 3000);
//         }
//     };
    
//     const convertMusicXML = (xmlData: string, toolkit: any = verovioToolkit) => {
//         if (!toolkit) {
//             showStatus('Verovio toolkit not initialized', 'error');
//             return;
//         }
        
//         try {
//             showStatus('Converting MusicXML to sheet music...', 'loading');
            
//             // Set Verovio options
//             const options = {
//                 scale: 100,
//                 pageWidth: 1600,
//                 pageHeight: 2100,
//                 adjustPageHeight: true,
//                 breaks: 'auto',
//                 font: 'Leipzig',
//                 header: 'none',
//                 footer: 'none'
//             };
            
//             toolkit.setOptions(options);
            
//             // Load the MusicXML data
//             const success = toolkit.loadData(xmlData);
            
//             if (!success) {
//                 throw new Error('Failed to load MusicXML data');
//             }
            
//             // Get number of pages
//             const pageCount = toolkit.getPageCount();
            
//             if (pageCount === 0) {
//                 throw new Error('No pages generated from MusicXML');
//             }
            
//             // Render all pages
//             let allSvg = '';
//             for (let i = 1; i <= pageCount; i++) {
//                 const pageSvg = toolkit.renderToSVG(i);
//                 if (pageSvg) {
//                     allSvg += pageSvg;
//                 }
//             }
            
//             if (!allSvg) {
//                 throw new Error('Failed to render SVG');
//             }
            
//             setCurrentSvg(allSvg);
//             showStatus(`Sheet music generated successfully! (${pageCount} page${pageCount > 1 ? 's' : ''})`, 'success');
            
//         } catch (error) {
//             console.error('Conversion error:', error);
//             showStatus(
//                 `Error converting MusicXML: ${
//                     typeof error === 'object' && error !== null && 'message' in error
//                         ? (error as { message: string }).message
//                         : String(error)
//                 }`,
//                 'error'
//             );
//         }
//     };
    
//     const downloadPNG = async () => {
//         if (!currentSvg) {
//             showStatus('No sheet music to download', 'error');
//             return;
//         }
        
//         try {
//             showStatus('Generating PNG...', 'loading');
            
//             const svgElement = document.querySelector('#sheetMusic svg');
//             if (!svgElement) {
//                 throw new Error('No SVG element found');
//             }
            
//             const canvas = await svgToCanvas(svgElement as SVGElement);
            
//             canvas.toBlob((blob) => {
//                 if (blob) {
//                     const url = URL.createObjectURL(blob);
//                     const a = document.createElement('a');
//                     a.href = url;
//                     a.download = `${currentFileName}_sheet_music.png`;
//                     a.click();
//                     URL.revokeObjectURL(url);
//                     showStatus('PNG downloaded successfully!', 'success');
//                 }
//             }, 'image/png');
            
//         } catch (error) {
//             console.error('PNG download error:', error);
//             showStatus(
//                 `Error downloading PNG: ${
//                     typeof error === 'object' && error !== null && 'message' in error
//                         ? (error as { message: string }).message
//                         : String(error)
//                 }`,
//                 'error'
//             );
//         }
//     };
    
//     const svgToCanvas = (svgElement: SVGElement): Promise<HTMLCanvasElement> => {
//         return new Promise((resolve, reject) => {
//             const canvas = document.createElement('canvas');
//             const ctx = canvas.getContext('2d');
            
//             if (!ctx) {
//                 reject(new Error('Cannot get 2D context'));
//                 return;
//             }
            
//             const viewBox = svgElement.getAttribute('viewBox');
//             let width = 800, height = 600;
            
//             if (viewBox) {
//                 const values = viewBox.split(' ');
//                 width = parseInt(values[2]);
//                 height = parseInt(values[3]);
//             } else {
//                 width = parseInt(svgElement.getAttribute('width') || '800');
//                 height = parseInt(svgElement.getAttribute('height') || '600');
//             }
            
//             canvas.width = width;
//             canvas.height = height;
            
//             const img = new Image();
//             img.onload = () => {
//                 ctx.fillStyle = 'white';
//                 ctx.fillRect(0, 0, canvas.width, canvas.height);
//                 ctx.drawImage(img, 0, 0);
//                 resolve(canvas);
//             };
            
//             img.onerror = () => reject(new Error('Failed to load SVG image'));
            
//             const svgData = new XMLSerializer().serializeToString(svgElement);
//             const svgBlob = new Blob([svgData], { type: 'image/svg+xml;charset=utf-8' });
//             const url = URL.createObjectURL(svgBlob);
//             img.src = url;
//         });
//     };
    
//     return (
//         <>
//             <Head>
//                 <title>Sheet Music Viewer</title>
//                 <script src="https://www.verovio.org/javascript/app/verovio-toolkit-wasm.js" />
//             </Head>
            
//             <div style={{ padding: '20px', maxWidth: '1200px', margin: '0 auto' }}>
//                 <h1>🎵 Sheet Music Viewer</h1>
                
//                 <div style={{ marginBottom: '20px' }}>
//                     <button 
//                         onClick={() => convertMusicXML(musicXMLData || '')}
//                         disabled={!isInitialized || !musicXMLData}
//                         style={{
//                             padding: '10px 20px',
//                             marginRight: '10px',
//                             backgroundColor: '#007bff',
//                             color: 'white',
//                             border: 'none',
//                             borderRadius: '5px',
//                             cursor: isInitialized && musicXMLData ? 'pointer' : 'not-allowed'
//                         }}
//                     >
//                         🎼 Convert to Sheet Music
//                     </button>
                    
//                     <button 
//                         onClick={downloadPNG}
//                         disabled={!currentSvg}
//                         style={{
//                             padding: '10px 20px',
//                             marginRight: '10px',
//                             backgroundColor: '#28a745',
//                             color: 'white',
//                             border: 'none',
//                             borderRadius: '5px',
//                             cursor: currentSvg ? 'pointer' : 'not-allowed'
//                         }}
//                     >
//                         📷 Download PNG
//                     </button>
                    
//                     <button 
//                         onClick={() => router.back()}
//                         style={{
//                             padding: '10px 20px',
//                             backgroundColor: '#6c757d',
//                             color: 'white',
//                             border: 'none',
//                             borderRadius: '5px',
//                             cursor: 'pointer'
//                         }}
//                     >
//                         ← Back
//                     </button>
//                 </div>
                
//                 {status.message && (
//                     <div style={{
//                         padding: '10px',
//                         marginBottom: '20px',
//                         backgroundColor: status.type === 'error' ? '#f8d7da' : 
//                                        status.type === 'success' ? '#d4edda' : '#d1ecf1',
//                         color: status.type === 'error' ? '#721c24' : 
//                                status.type === 'success' ? '#155724' : '#0c5460',
//                         border: '1px solid',
//                         borderColor: status.type === 'error' ? '#f5c6cb' : 
//                                    status.type === 'success' ? '#c3e6cb' : '#bee5eb',
//                         borderRadius: '5px'
//                     }}>
//                         {status.message}
//                     </div>
//                 )}
                
//                 <div 
//                     id="sheetMusic" 
//                     style={{
//                         border: '1px solid #ddd',
//                         borderRadius: '5px',
//                         padding: '20px',
//                         backgroundColor: 'white',
//                         minHeight: '400px'
//                     }}
//                     dangerouslySetInnerHTML={{ __html: currentSvg }}
//                 />
//             </div>
//         </>
//     );
// };

// export default SheetMusicPage;