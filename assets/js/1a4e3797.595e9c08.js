"use strict";(self.webpackChunkbrick_bar=self.webpackChunkbrick_bar||[]).push([[7920],{6060:(e,t,a)=>{a.r(t),a.d(t,{default:()=>S});var n=a(7294),r=a(9962),c=a(9185),s=a(1514),l=a(3699),u=a(6550),o=a(6136);const m=function(){const e=(0,u.k6)(),t=(0,u.TH)(),{siteConfig:{baseUrl:a}}=(0,r.Z)();return{searchValue:o.Z.canUseDOM&&new URLSearchParams(t.search).get("q")||"",updateSearchPath:a=>{const n=new URLSearchParams(t.search);a?n.set("q",a):n.delete("q"),e.replace({search:n.toString()})},generateSearchPageLink:e=>`${a}search?q=${encodeURIComponent(e)}`}};var h=a(5202),i=a(6654),p=a(1523),d=a(7976),f=a(9395),I=a(4056),_=a(5901);function E(e,t){return e.replace(/\{\{\s*(\w+)\s*\}\}/g,((e,a)=>Object.prototype.hasOwnProperty.call(t,a)?String(t[a]):e))}const g={searchQueryInput:"searchQueryInput_CFBF",searchResultItem:"searchResultItem_U687",searchResultItemPath:"searchResultItemPath_uIbk",searchResultItemSummary:"searchResultItemSummary_oZHr"};function y(e){let{searchResult:{document:t,type:a,page:r,tokens:c,metadata:s}}=e;const u=0===a,o=2===a,m=(u?t.b:r.b).slice(),h=o?t.s:t.t;return u||m.push(r.t),n.createElement("article",{className:g.searchResultItem},n.createElement("h2",null,n.createElement(l.Z,{to:t.u+(t.h||""),dangerouslySetInnerHTML:{__html:o?(0,p.C)(h,c):(0,d.o)(h,(0,f.m)(s,"t"),c,100)}})),m.length>0&&n.createElement("p",{className:g.searchResultItemPath},m.join(" \u203a ")),o&&n.createElement("p",{className:g.searchResultItemSummary,dangerouslySetInnerHTML:{__html:(0,d.o)(t.t,(0,f.m)(s,"t"),c,100)}}))}const S=function(){const{siteConfig:{baseUrl:e}}=(0,r.Z)(),{searchValue:t,updateSearchPath:a}=m(),[l,u]=(0,n.useState)(t),[o,p]=(0,n.useState)(),[d,f]=(0,n.useState)(),S=(0,n.useMemo)((()=>l?E(_.Iz.search_results_for,{keyword:l}):_.Iz.search_the_documentation),[l]);(0,n.useEffect)((()=>{a(l),o&&(l?o(l,(e=>{f(e)})):f(void 0))}),[l,o]);const b=(0,n.useCallback)((e=>{u(e.target.value)}),[]);return(0,n.useEffect)((()=>{t&&t!==l&&u(t)}),[t]),(0,n.useEffect)((()=>{!async function(){const{wrappedIndexes:t,zhDictionary:a}=await(0,h.w)(e);p((()=>(0,i.v)(t,a,100)))}()}),[e]),n.createElement(c.Z,{title:S},n.createElement(s.Z,null,n.createElement("meta",{property:"robots",content:"noindex, follow"})),n.createElement("div",{className:"container margin-vert--lg"},n.createElement("h1",null,S),n.createElement("input",{type:"search",name:"q",className:g.searchQueryInput,"aria-label":"Search",onChange:b,value:l,autoComplete:"off",autoFocus:!0}),!o&&l&&n.createElement("div",null,n.createElement(I.Z,null)),d&&(d.length>0?n.createElement("p",null,E(1===d.length?_.Iz.count_documents_found:_.Iz.count_documents_found_plural,{count:d.length})):n.createElement("p",null,_.Iz.no_documents_were_found)),n.createElement("section",null,d&&d.map((e=>n.createElement(y,{key:e.document.i,searchResult:e}))))))}}}]);