"use client"
import React, { useState, useEffect } from 'react';
// Interfaces for the academic content
interface AcademicContent {
    abstract: string;
    sections: Section[];
  }
  
  interface Section {
    title: string;
    content: string;
    image?: {
      alt: string;
      src: string;
    };
  }
  
export function AcademicPage() {
  const [data, setData] = useState<AcademicContent | null>(null);

  useEffect(() => {
    // Fetching the data from a local JSON file
    fetch('research-paper-content.json')
      .then(response => response.json())
      .then(setData)
      .catch(console.error); // Handle errors appropriately in real applications
  }, []);

  if (!data) {
    return <div>Loading...</div>; // Or any other loading state representation
  }

  return (
    <>
      <section className="container py-12 md:py-20">
        <div className="max-w-3xl mx-auto space-y-8">
          <div>
            <h2 className="text-2xl font-bold mb-4">Abstract</h2>
            <p className="text-gray-500 dark:text-gray-400 leading-relaxed">
              {data.abstract}
            </p>
          </div>
          {data.sections.map((section, index) => (
            <div key={index}>
              <h2 className="text-2xl font-bold mb-4">{section.title}</h2>
              <p className="text-gray-500 dark:text-gray-400 leading-relaxed">
                {section.content}
              </p>
              {section.image && (
                <div className="mt-4">
                  <img
                    alt={section.image.alt}
                    className="rounded-lg"
                    style={{
                      aspectRatio: "800/450",
                      objectFit: "cover",
                    }}
                    src={section.image.src}
                    width={800}
                    height={450}
                  />
                  <p className="text-sm text-gray-500 dark:text-gray-400 mt-2">
                    {section.image.alt}
                  </p>
                </div>
              )}
            </div>
          ))}
        </div>
      </section>
    </>
  );
}
