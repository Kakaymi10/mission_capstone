import React, { useRef, useState, useEffect } from 'react';
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome';
import {
  faCamera,
  faMagnifyingGlass,
  faDrawPolygon,
  faPen,
  faHighlighter,
  faFileExport,
  faExpand,
  faRotate,
  faRobot,
  faPaperPlane,
  faPlayCircle,
  faFileImport,
  faTable,
  faBookmark,
  faHistory,
  faMicrophone,
  faUpload,
} from '@fortawesome/free-solid-svg-icons';

const App = () => {
  const canvasRef = useRef(null);
  const [image, setImage] = useState(null);
  const [isDrawing, setIsDrawing] = useState(false);
  const [tool, setTool] = useState('pen'); // 'pen', 'highlighter', 'polygon'
  const [lineWidth, setLineWidth] = useState(2);
  const [lineColor, setLineColor] = useState('#000000');

  // Load the initial image
  useEffect(() => {
    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');

    const img = new Image();
    img.src = 'https://storage.googleapis.com/uxpilot-auth.appspot.com/79891f72a3-5d8297d12e6249d91caf.png';
    img.onload = () => {
      setImage(img);
      ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
    };
  }, []);

  // Handle image upload
  const handleImageUpload = (e) => {
    const file = e.target.files[0];
    if (file) {
      const reader = new FileReader();
      reader.onload = (event) => {
        const img = new Image();
        img.src = event.target.result;
        img.onload = () => {
          setImage(img);
          const canvas = canvasRef.current;
          const ctx = canvas.getContext('2d');
          ctx.clearRect(0, 0, canvas.width, canvas.height);
          ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
        };
      };
      reader.readAsDataURL(file);
    }
  };

  // Drawing functionality
  const startDrawing = (e) => {
    if (tool === 'pen' || tool === 'highlighter') {
      setIsDrawing(true);
      const canvas = canvasRef.current;
      const ctx = canvas.getContext('2d');
      ctx.beginPath();
      ctx.moveTo(e.nativeEvent.offsetX, e.nativeEvent.offsetY);
    }
  };

  const draw = (e) => {
    if (!isDrawing) return;
    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');
    ctx.lineWidth = tool === 'highlighter' ? 10 : lineWidth;
    ctx.strokeStyle = tool === 'highlighter' ? 'rgba(255, 255, 0, 0.5)' : lineColor;
    ctx.lineTo(e.nativeEvent.offsetX, e.nativeEvent.offsetY);
    ctx.stroke();
  };

  const stopDrawing = () => {
    setIsDrawing(false);
  };

  // Clear canvas
  const clearCanvas = () => {
    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    if (image) {
      ctx.drawImage(image, 0, 0, canvas.width, canvas.height);
    }
  };

  return (
    <div className="h-full text-base-content">
      <div id="root" className="flex h-full bg-gray-100">
        {/* Microscope Panel */}
        <div id="microscope-panel" className="w-3/5 h-full bg-white p-4 border-r border-gray-200">
          <div id="microscope-controls" className="flex items-center justify-between mb-4">
            <div className="flex items-center space-x-4">
              <button className="bg-blue-600 text-white px-4 py-2 rounded-lg flex items-center">
                <FontAwesomeIcon icon={faCamera} className="mr-2" />
                Capture
              </button>
              <label className="bg-blue-600 text-white px-4 py-2 rounded-lg flex items-center cursor-pointer">
                <FontAwesomeIcon icon={faUpload} className="mr-2" />
                Upload Slide
                <input type="file" accept="image/*" className="hidden" onChange={handleImageUpload} />
              </label>
              <div className="flex items-center bg-gray-100 rounded-lg px-3 py-2">
                <FontAwesomeIcon icon={faMagnifyingGlass} className="mr-2" />
                <input type="range" className="w-32" min="1" max="100" value="50" />
                <span className="ml-2 text-sm text-gray-600">50x</span>
              </div>
            </div>
            <div className="flex space-x-3">
              <button
                className={`p-2 hover:bg-gray-100 rounded-lg ${tool === 'pen' ? 'bg-gray-200' : ''}`}
                onClick={() => setTool('pen')}
              >
                <FontAwesomeIcon icon={faPen} />
              </button>
              <button
                className={`p-2 hover:bg-gray-100 rounded-lg ${tool === 'highlighter' ? 'bg-gray-200' : ''}`}
                onClick={() => setTool('highlighter')}
              >
                <FontAwesomeIcon icon={faHighlighter} />
              </button>
              <button
                className={`p-2 hover:bg-gray-100 rounded-lg ${tool === 'polygon' ? 'bg-gray-200' : ''}`}
                onClick={() => setTool('polygon')}
              >
                <FontAwesomeIcon icon={faDrawPolygon} />
              </button>
              <button className="p-2 hover:bg-gray-100 rounded-lg" onClick={clearCanvas}>
                <FontAwesomeIcon icon={faFileExport} />
              </button>
            </div>
          </div>
          <div id="microscope-view" className="relative h-[600px] bg-gray-900 rounded-lg overflow-hidden">
            <canvas
              ref={canvasRef}
              width={800}
              height={600}
              className="w-full h-full"
              onMouseDown={startDrawing}
              onMouseMove={draw}
              onMouseUp={stopDrawing}
              onMouseLeave={stopDrawing}
            />
            <div className="absolute bottom-0 left-0 right-0 bg-gradient-to-t from-black/70 to-transparent p-4">
              <div className="flex items-center justify-between text-white">
                <div>
                  <span className="text-sm opacity-80">Current View:</span>
                  <h3 className="font-semibold">Human Blood Cell Sample</h3>
                </div>
                <div className="flex items-center space-x-2">
                  <button className="p-2 hover:bg-white/10 rounded">
                    <FontAwesomeIcon icon={faExpand} />
                  </button>
                  <button className="p-2 hover:bg-white/10 rounded">
                    <FontAwesomeIcon icon={faRotate} />
                  </button>
                </div>
              </div>
            </div>
          </div>
          <div id="detection-info" className="mt-4 p-4 bg-gray-50 rounded-lg">
            <h4 className="font-semibold mb-2">AI Detection Results</h4>
            <div className="flex flex-wrap gap-2">
              <span className="px-3 py-1 bg-blue-100 text-blue-700 rounded-full text-sm">Red Blood Cells (15)</span>
              <span className="px-3 py-1 bg-green-100 text-green-700 rounded-full text-sm">White Blood Cells (3)</span>
              <span className="px-3 py-1 bg-purple-100 text-purple-700 rounded-full text-sm">Platelets (8)</span>
            </div>
          </div>
        </div>

        {/* AI Panel */}
        <div id="ai-panel" className="w-2/5 h-full bg-white p-4">
          
        </div>
      </div>
    </div>
  );
};

export default App;