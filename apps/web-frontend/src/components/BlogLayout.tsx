"use client";

import React, { ReactNode, useEffect, useState } from "react";
import { BlogHeading, BlogTableOfContents } from "./BlogHeading";

interface BlogLayoutProps {
  children: ReactNode;
  title: string;
  date: string;
  showTableOfContents?: boolean;
}

/**
 * A layout component for blog posts that automatically:
 * 1. Adds heading anchors with hash links
 * 2. Generates a table of contents
 */
export default function BlogLayout({
  children,
  title,
  date,
  showTableOfContents = true,
}: BlogLayoutProps) {
  const [headings, setHeadings] = useState<{ text: string; level: number }[]>(
    []
  );

  // Scan the article for headings after render to build the table of contents
  useEffect(() => {
    const articleElement = document.querySelector("article");
    if (!articleElement) return;

    const headingElements =
      articleElement.querySelectorAll("h2, h3, h4, h5, h6");
    const extractedHeadings = Array.from(headingElements).map((el) => ({
      text: el.textContent || "",
      level: parseInt(el.tagName.charAt(1)),
    }));

    setHeadings(extractedHeadings);
  }, []);

  // Recursively transform h2-h6 tags to BlogHeading components
  const recursivelyReplaceHeadings = (children: ReactNode): ReactNode => {
    return React.Children.map(children, (child) => {
      if (!React.isValidElement(child)) {
        return child;
      }

      const { type, props } = child;

      // Transform heading elements
      if (typeof type === "string" && /^h[2-6]$/.test(type)) {
        const level = parseInt(type.charAt(1)) as 1 | 2 | 3 | 4 | 5 | 6;
        return (
          <BlogHeading level={level} className={props.className || ""}>
            {props.children}
          </BlogHeading>
        );
      }

      // Recursively process children if they exist
      if (props && props.children) {
        const newChildren = recursivelyReplaceHeadings(props.children);
        return React.cloneElement(child, { ...props, children: newChildren });
      }

      return child;
    });
  };

  // Process the children to replace headings
  const processedChildren = recursivelyReplaceHeadings(children);

  return (
    <main className="flex min-h-screen w-full flex-col items-center bg-background text-foreground">
      {/* Header Section */}
      <div className="w-full bg-gradient-to-b from-primary/10 to-background py-8">
        <div className="container flex flex-col items-center justify-center gap-4 px-4">
          <h1 className="text-4xl font-bold tracking-tight text-primary text-center">
            {title}
          </h1>
          <div className="flex items-center gap-2 text-sm text-muted-foreground">
            <time dateTime={date}>{date}</time>
          </div>
        </div>
      </div>

      {/* Article Content */}
      <article className="container px-4 pb-12 prose prose-invert prose-headings:mb-3 prose-headings:mt-6 max-w-3xl">
        {showTableOfContents && headings.length > 0 && (
          <BlogTableOfContents headings={headings} />
        )}
        {processedChildren}
      </article>
    </main>
  );
}
